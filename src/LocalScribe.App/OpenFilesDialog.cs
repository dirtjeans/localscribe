using System.Runtime.InteropServices;

namespace LocalScribe.App;

/// <summary>
/// The classic Windows Open dialog, through its COM interface, for one file or several.
/// <para>
/// Not <c>Windows.Storage.Pickers.FileOpenPicker</c>: on Windows 11 that picker runs in a
/// separate process (pickerhost.exe), and for this unpackaged app its multiple-file pick closed
/// without ever returning — no files, no error, no exception. This dialog runs in our own
/// process and hands back plain paths, which is all the app wants anyway.
/// </para>
/// </summary>
internal static class OpenFilesDialog
{
    /// <summary>The chosen paths, empty if the dialog was cancelled.</summary>
    public static IReadOnlyList<string> Show(nint owner, string title, string filterName, IEnumerable<string> extensions)
    {
        var dialog = (IFileOpenDialog)new FileOpenDialog();

        try
        {
            dialog.SetTitle(title);
            dialog.SetOptions(AllowMultiSelect | FileMustExist | PathMustExist | ForceFileSystem);
            dialog.SetFileTypes(2,
            [
                new FilterSpec { Name = filterName, Spec = string.Join(';', extensions.Select(e => "*" + e)) },
                new FilterSpec { Name = "All files", Spec = "*.*" },
            ]);

            if (dialog.Show(owner) != 0)
            {
                // Cancelled, or the dialog could not open; either way nothing was chosen.
                return [];
            }

            dialog.GetResults(out var results);
            results.GetCount(out var count);

            var paths = new List<string>((int)count);
            for (uint i = 0; i < count; i++)
            {
                results.GetItemAt(i, out var item);
                item.GetDisplayName(FileSystemPath, out var buffer);

                try
                {
                    if (Marshal.PtrToStringUni(buffer) is { Length: > 0 } path)
                    {
                        paths.Add(path);
                    }
                }
                finally
                {
                    Marshal.FreeCoTaskMem(buffer);
                }
            }

            return paths;
        }
        finally
        {
            Marshal.ReleaseComObject(dialog);
        }
    }

    private const uint ForceFileSystem = 0x40;
    private const uint AllowMultiSelect = 0x200;
    private const uint PathMustExist = 0x800;
    private const uint FileMustExist = 0x1000;
    private const uint FileSystemPath = 0x80058000;   // SIGDN_FILESYSPATH

    [ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    private class FileOpenDialog;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct FilterSpec
    {
        [MarshalAs(UnmanagedType.LPWStr)] public string Name;
        [MarshalAs(UnmanagedType.LPWStr)] public string Spec;
    }

    // Declared in vtable order; the methods the app never calls are kept as placeholders,
    // because COM finds each method by its position, not its name.
    [ComImport, Guid("D57C7288-D4AD-4768-BE02-9D969532D960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileOpenDialog
    {
        [PreserveSig] int Show(nint parent);
        void SetFileTypes(uint count, [MarshalAs(UnmanagedType.LPArray)] FilterSpec[] filters);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise(nint events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(uint options);
        void GetOptions(out uint options);
        void SetDefaultFolder(IShellItem item);
        void SetFolder(IShellItem item);
        void GetFolder(out IShellItem item);
        void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem item);
        void AddPlace(IShellItem item, int placement);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int result);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(nint filter);
        void GetResults(out IShellItemArray items);
        void GetSelectedItems(out IShellItemArray items);
    }

    [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem
    {
        void BindToHandler(nint context, ref Guid handler, ref Guid interfaceId, out nint result);
        void GetParent(out IShellItem parent);
        void GetDisplayName(uint form, out nint name);
        void GetAttributes(uint mask, out uint attributes);
        void Compare(IShellItem other, uint hint, out int order);
    }

    [ComImport, Guid("B63EA76D-1F85-456F-A19C-48159EFA858B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItemArray
    {
        void BindToHandler(nint context, ref Guid handler, ref Guid interfaceId, out nint result);
        void GetPropertyStore(int flags, ref Guid interfaceId, out nint store);
        void GetPropertyDescriptionList(nint key, ref Guid interfaceId, out nint list);
        void GetAttributes(int how, uint mask, out uint attributes);
        void GetCount(out uint count);
        void GetItemAt(uint index, out IShellItem item);
        void EnumItems(out nint items);
    }
}
