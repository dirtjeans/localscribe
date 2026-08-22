using LocalScribe.Core.Refinement;
using Xunit;

namespace LocalScribe.Core.Tests;

public class CleanupReportTests
{
    /// <summary>Silence is the right answer when there is nothing wrong.</summary>
    [Fact]
    public void NothingWentWrongIsNotWorthSaying() =>
        Assert.Null(CleanupReport.Describe(failed: 0, rejected: 0));

    /// <summary>
    /// A call that never came back is usually the backend rather than the recording, and is worth
    /// trying again. The wording has to say so, because the reader's next move depends on it.
    /// </summary>
    [Fact]
    public void ACallThatFailedSaysTheModelDidNotAnswer()
    {
        var notice = CleanupReport.Describe(failed: 3, rejected: 0);

        Assert.NotNull(notice);
        Assert.Contains("3 passages were", notice, StringComparison.Ordinal);
        Assert.Contains("did not answer", notice, StringComparison.Ordinal);
    }

    /// <summary>An answer that did not match is a different fault and gets different words.</summary>
    [Fact]
    public void ARejectedRewriteSaysTheAnswerDidNotMatch()
    {
        var notice = CleanupReport.Describe(failed: 0, rejected: 2);

        Assert.NotNull(notice);
        Assert.Contains("did not match what was said", notice, StringComparison.Ordinal);
        Assert.DoesNotContain("did not answer", notice, StringComparison.Ordinal);
    }

    [Fact]
    public void BothFaultsAreCountedSeparately()
    {
        var notice = CleanupReport.Describe(failed: 2, rejected: 3);

        Assert.NotNull(notice);
        Assert.Contains("5 passages were", notice, StringComparison.Ordinal);
        Assert.Contains("for 2", notice, StringComparison.Ordinal);
        Assert.Contains("for 3", notice, StringComparison.Ordinal);
    }

    [Fact]
    public void OneIsNotWrittenAsAPlural()
    {
        var notice = CleanupReport.Describe(failed: 1, rejected: 0);

        Assert.NotNull(notice);
        Assert.StartsWith("One passage was", notice, StringComparison.Ordinal);
    }

    /// <summary>The reason names the problem, which is most of what makes it fixable.</summary>
    [Fact]
    public void TheReasonIsCarriedThrough()
    {
        var notice = CleanupReport.Describe(failed: 1, rejected: 0, "Connection refused");

        Assert.NotNull(notice);
        Assert.Contains("(Connection refused)", notice, StringComparison.Ordinal);
    }

    /// <summary>
    /// A rejection has no error of its own. Attaching one left over from some earlier failure
    /// would name the wrong problem, which is worse than naming none.
    /// </summary>
    [Fact]
    public void ARejectionIsNotGivenSomeoneElsesReason()
    {
        var notice = CleanupReport.Describe(failed: 0, rejected: 2, "Connection refused");

        Assert.NotNull(notice);
        Assert.DoesNotContain("Connection refused", notice, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmptyReasonIsNotShownAsEmptyBrackets()
    {
        var notice = CleanupReport.Describe(failed: 1, rejected: 0, string.Empty);

        Assert.NotNull(notice);
        Assert.DoesNotContain("(", notice, StringComparison.Ordinal);
    }
}
