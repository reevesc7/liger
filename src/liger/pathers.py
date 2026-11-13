from datetime import datetime, timezone
from pathlib import Path
import time


def datetime_subpath(
    dir: str | Path,
    name: str = "{}",
    datetime_fmt: str = "%Y-%m-%d_%H-%M-%S.%f",
    time_zone: timezone | None = timezone.utc,
    absolute: bool = False,
) -> Path:
    """Make a path which represents a child path of the given path, with
    the current date and time included in its name.

    Parameters
    ----------
    dir : str | Path
        The path of the parent directory of the result path.
        May be relative or absolute.
    name : str, default "{}"
        The name of the result path. If this includes "{}", the date and time
        is substituted at that location. Otherwise, date and time is added to
        the end of the name.
    datetime_fmt : str, default "%Y-%m-%d_%H-%M-%S.%f"
        Format of the date and time string representation.
    time_zone : timezone | None, default timezone.utc
        Time zone of the date and time. If None (which is not the default and
        must be passed), the local time zone is used.
    absolute : bool, default False
        If True, an absolute path is always returned. If dir was already absolute,
        this has no effect.

    Returns
    -------
    subpath : Path
        Child path
    """
    dir = Path(dir)
    if "{}" not in name:
        name += "{}"
    for _ in range(100):
        now = datetime.now(time_zone).strftime(datetime_fmt)
        subpath = (dir / name.format(now))
        if not subpath.exists():
            break
        time.sleep(0.01)
    else:
        raise RuntimeError(f"Could not resolve a non-existing path like {dir / name}")
    if absolute:
        return subpath.resolve()
    return subpath
