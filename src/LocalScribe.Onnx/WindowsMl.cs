using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace LocalScribe.Onnx;

/// <summary>
/// The hardware providers ONNX Runtime gets from Windows ML on Windows on ARM: Qualcomm's for the
/// NPU, fetched through Windows ML's catalog, and DirectML for the GPU, built in.
/// <para>
/// Qualcomm's provider is not in the app. Windows ML keeps a catalog of providers Microsoft has
/// certified for this machine, installs them as system packages, and hands each process the
/// library to register. Fetching is a one-time download from Microsoft, made by the doctor's
/// <c>--fetch-models</c> or the app's first run, never on a launch that already has it.
/// </para>
/// <para>
/// The catalog is reached through Windows ML's flat C API rather than its WinRT projection,
/// which would oblige every project that touches ONNX Runtime to target a Windows framework.
/// </para>
/// </summary>
public static class WindowsMl
{
    /// <summary>The name ONNX Runtime and the catalog both give Qualcomm's provider.</summary>
    public const string QnnProvider = "QNNExecutionProvider";

    /// <summary>The name of the built-in DirectML provider.</summary>
    public const string DirectMlProvider = "DmlExecutionProvider";

    private static readonly object Gate = new();
    private static ProviderState? _state;

    /// <summary>Whether this build's ONNX Runtime comes from Windows ML.</summary>
    public static bool InUse =>
#if WINDOWS_ML
        true;
#else
        false;
#endif

    /// <summary>What the catalog had, what was registered, and why anything was not.</summary>
    /// <param name="NpuRegistered">Qualcomm's provider is registered with ONNX Runtime.</param>
    /// <param name="NpuMissing">The catalog offers it but it has not been downloaded.</param>
    /// <param name="Detail">One line for a person: versions, or what went wrong.</param>
    public sealed record ProviderState(bool NpuRegistered, bool NpuMissing, string Detail);

    /// <summary>
    /// Registers the certified providers already on this machine, once per process, and when
    /// <paramref name="allowDownload"/> is set fetches any the catalog offers but the machine
    /// lacks. Without it nothing reaches the network: a provider not yet fetched is reported
    /// missing and the planner places the work elsewhere.
    /// </summary>
    public static ProviderState Prepare(bool allowDownload = false, Action<string>? log = null)
    {
        lock (Gate)
        {
            if (_state is { } done && (done.NpuRegistered || !done.NpuMissing || !allowDownload))
            {
                return done;
            }

#if WINDOWS_ML
            // Off the caller's thread when that is a single-threaded apartment, a window's UI
            // thread: there the catalog refused to ready a provider it readied everywhere else
            // (0x80070015, not ready), and the first ask came from the window.
            _state = Thread.CurrentThread.GetApartmentState() == ApartmentState.STA
                ? Task.Run(() => PrepareCore(allowDownload, log)).GetAwaiter().GetResult()
                : PrepareCore(allowDownload, log);
#else
            _state = new ProviderState(false, false, "this build does not use Windows ML");
#endif
            return _state;
        }
    }

    /// <summary>The devices one provider offers, of one kind, after <see cref="Prepare"/>.</summary>
    public static IReadOnlyList<OrtEpDevice> Devices(string provider, OrtHardwareDeviceType type)
    {
        Prepare();

        try
        {
            return [.. OrtEnv.Instance().GetEpDevices().Where(d => d.EpName == provider && d.HardwareDevice.Type == type)];
        }
        catch (Exception exception) when (exception is OnnxRuntimeException or EntryPointNotFoundException or DllNotFoundException)
        {
            return [];
        }
    }

    /// <summary>
    /// Adds Qualcomm's provider for one of its backends, NPU or GPU, or throws saying why not.
    /// <para>
    /// Where ONNX Runtime came from the QNN package, the backend is chosen by naming its library;
    /// from Windows ML it is chosen by device, and naming a library is an error.
    /// </para>
    /// </summary>
    public static void AppendQnn(
        SessionOptions options,
        IReadOnlyDictionary<string, string> providerOptions,
        OrtHardwareDeviceType type = OrtHardwareDeviceType.NPU)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(providerOptions);

#if WINDOWS_ML
        var devices = Devices(QnnProvider, type);
        if (devices.Count == 0)
        {
            throw new InvalidOperationException(
                $"Windows ML has no Qualcomm provider for the {type}: {Prepare().Detail}. "
                + "Run 'localscribe-doctor --fetch-models' to fetch it.");
        }

        var settings = providerOptions
            .Where(pair => pair.Key != "backend_path")
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        options.AppendExecutionProvider(OrtEnv.Instance(), devices, settings);
#else
        var settings = new Dictionary<string, string>(providerOptions)
        {
            ["backend_path"] = type == OrtHardwareDeviceType.GPU ? "QnnGpu.dll" : "QnnHtp.dll",
        };
        options.AppendExecutionProvider("QNN", settings);
#endif
    }

    /// <summary>
    /// Adds DirectML. It runs one graph at a time and plans its own memory, so ONNX Runtime's
    /// memory pattern and parallel execution are off for its sessions, as DirectML requires.
    /// </summary>
    public static void AppendDirectMl(SessionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        options.EnableMemoryPattern = false;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

#if WINDOWS_ML
        var devices = Devices(DirectMlProvider, OrtHardwareDeviceType.GPU);
        if (devices.Count > 0)
        {
            options.AppendExecutionProvider(OrtEnv.Instance(), devices, new Dictionary<string, string>());
            return;
        }
#endif
        options.AppendExecutionProvider_DML(deviceId: 0);
    }

#if WINDOWS_ML
    // The catalog lives for the process: readying a provider ties its package to this process,
    // and releasing the catalog is not documented to leave that in place. Its provider handles
    // live as long as it does.
    private static nint _catalog;

    private static ProviderState PrepareCore(bool allowDownload, Action<string>? log)
    {
        try
        {
            if (_catalog == 0)
            {
                Check(Native.WinMLEpCatalogCreate(out _catalog), "opening the catalog");
            }

            // Handles only. The info the enumeration passes alongside came back with its paths
            // null and its fields shifting between passes (2.1.74), and a lookup by name can
            // return the uncertified copy, so everything is read from each provider's handle.
            var handles = new List<nint>();
            Native.EnumCallback collect = (handle, _, _) =>
            {
                handles.Add(handle);
                return true;
            };

            Check(Native.WinMLEpCatalogEnumProviders(_catalog, collect, 0), "listing providers");
            GC.KeepAlive(collect);

            var registered = new List<string>();
            var missing = new List<string>();
            var env = OrtEnv.Instance();
            var already = env.GetEpDevices().Select(d => d.EpName).ToHashSet();

            foreach (var handle in handles)
            {
                // Uncertified providers are the ones Windows installs for its own features: the
                // catalog lists them, and its own "register certified" passes them over.
                Check(Native.WinMLEpGetCertification(handle, out var certification), "reading a certification");
                if (certification != Native.Certification.Certified)
                {
                    continue;
                }

                var name = Read(handle, Native.WinMLEpGetNameSize, Native.WinMLEpGetName, "a name");
                Check(Native.WinMLEpGetReadyState(handle, out var state), $"reading {name}'s state");

                if (state == Native.ReadyState.NotPresent && !allowDownload)
                {
                    missing.Add(name);
                    continue;
                }

                if (state == Native.ReadyState.NotPresent)
                {
                    log?.Invoke($"fetching {name} from Microsoft");
                }

                // Installed already, this only ties its package to the process; the library
                // path is empty until it has.
                Check(Native.WinMLEpEnsureReady(handle), $"readying {name}");
                var library = Read(handle, Native.WinMLEpGetLibraryPathSize, Native.WinMLEpGetLibraryPath, $"{name}'s library");

                if (!already.Contains(name))
                {
                    try
                    {
                        env.RegisterExecutionProviderLibrary(name, library);
                    }
                    catch (OnnxRuntimeException exception)
                    {
                        throw new InvalidOperationException($"registering {library}: {exception.Message}", exception);
                    }
                }

                var version = Read(handle, Native.WinMLEpGetVersionSize, Native.WinMLEpGetVersion, $"{name}'s version");
                registered.Add($"{name} {version}".Trim());
            }

            var npu = registered.Any(r => r.StartsWith(QnnProvider, StringComparison.Ordinal));
            var npuMissing = missing.Contains(QnnProvider);
            var detail = string.Join("; ", [
                .. registered.Select(r => $"registered {r}"),
                .. missing.Select(m => $"{m} not yet fetched"),
            ]);

            return new ProviderState(npu, npuMissing, detail.Length > 0 ? detail : "the catalog offers no certified provider here");
        }
        catch (Exception exception) when (exception is COMException or DllNotFoundException or EntryPointNotFoundException or OnnxRuntimeException or InvalidOperationException)
        {
            return new ProviderState(false, false, $"Windows ML: {exception.Message}");
        }
    }

    private static string Read(nint handle, Native.SizeOf sizeOf, Native.ValueOf valueOf, string what)
    {
        Check(sizeOf(handle, out var size), $"reading {what}");
        var buffer = new byte[(int)size];
        Check(valueOf(handle, size, buffer, out _), $"reading {what}");
        return System.Text.Encoding.UTF8.GetString(buffer).TrimEnd('\0');
    }

    private static void Check(int hresult, string doing)
    {
        if (hresult < 0)
        {
            throw new COMException($"Windows ML failed {doing} (0x{hresult:X8})", hresult);
        }
    }

    private static class Native
    {
        private const string Library = "Microsoft.Windows.AI.MachineLearning";

        public enum ReadyState
        {
            Ready = 0,
            NotReady = 1,
            NotPresent = 2,
        }

        public enum Certification
        {
            Unknown = 0,
            Certified = 1,
            Uncertified = 2,
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public delegate bool EnumCallback(nint ep, nint info, nint context);

        public delegate int SizeOf(nint ep, out nuint size);

        public delegate int ValueOf(nint ep, nuint bufferSize, byte[] buffer, out nuint used);

        [DllImport(Library)]
        public static extern int WinMLEpCatalogCreate(out nint catalog);

        [DllImport(Library)]
        public static extern int WinMLEpCatalogEnumProviders(nint catalog, EnumCallback callback, nint context);

        [DllImport(Library)]
        public static extern int WinMLEpGetCertification(nint ep, out Certification certification);

        [DllImport(Library)]
        public static extern int WinMLEpGetReadyState(nint ep, out ReadyState state);

        [DllImport(Library)]
        public static extern int WinMLEpEnsureReady(nint ep);

        [DllImport(Library)]
        public static extern int WinMLEpGetNameSize(nint ep, out nuint size);

        [DllImport(Library)]
        public static extern int WinMLEpGetName(nint ep, nuint bufferSize, byte[] buffer, out nuint used);

        [DllImport(Library)]
        public static extern int WinMLEpGetVersionSize(nint ep, out nuint size);

        [DllImport(Library)]
        public static extern int WinMLEpGetVersion(nint ep, nuint bufferSize, byte[] buffer, out nuint used);

        [DllImport(Library)]
        public static extern int WinMLEpGetLibraryPathSize(nint ep, out nuint size);

        [DllImport(Library)]
        public static extern int WinMLEpGetLibraryPath(nint ep, nuint bufferSize, byte[] buffer, out nuint used);
    }
#endif
}
