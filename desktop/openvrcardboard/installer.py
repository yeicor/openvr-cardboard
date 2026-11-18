"""Command-line installer for the portable OpenVR Cardboard build.

``openvrcardboard-install`` copies the SteamVR driver into the detected SteamVR
installation and registers the bundled Monado runtime as the user's active
OpenXR runtime. ``openvrcardboard-install uninstall`` restores every file that
was replaced and removes everything the installer added.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .integration import (
    DEFAULT_OPENXR_REMOTE_PORT,
    DEFAULT_OPENXR_VIEW_HEIGHT,
    DEFAULT_OPENXR_VIEW_WIDTH,
    IntegrationError,
    bundled_paths,
    find_steamvr,
    install,
    status,
    uninstall,
)


def _print_state(state) -> None:
    if not state.installed:
        print("OpenVR Cardboard integration is not installed.")
        return
    print(json.dumps(asdict(state), indent=2, sort_keys=True))


def _default_driver_source() -> Optional[Path]:
    driver, _ = bundled_paths()
    return driver


def _default_openxr_prefix() -> Optional[Path]:
    _, prefix = bundled_paths()
    return prefix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openvrcardboard-install",
        description=(
            "Install or remove the SteamVR driver and the bundled OpenXR "
            "runtime integration."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="install",
        choices=("install", "uninstall", "status"),
        help="action to perform (default: install)",
    )
    parser.add_argument(
        "--steamvr",
        metavar="PATH",
        help="SteamVR root; auto-detected when omitted",
    )
    parser.add_argument(
        "--driver",
        metavar="PATH",
        help="driver folder to install (default: bundled driver)",
    )
    parser.add_argument(
        "--openxr",
        metavar="PATH",
        help="OpenXR runtime prefix (default: bundled runtime)",
    )
    parser.add_argument(
        "--no-driver",
        action="store_true",
        help="do not install the SteamVR driver",
    )
    parser.add_argument(
        "--no-openxr",
        action="store_true",
        help="do not register the OpenXR runtime",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files that no longer look like ours",
    )
    parser.add_argument(
        "--state",
        metavar="PATH",
        help="integration state file (default: next to the app config)",
    )
    parser.add_argument(
        "--config-home",
        metavar="PATH",
        help="XDG config home used for OpenXR and Monado registration",
    )
    parser.add_argument(
        "--view-width",
        type=int,
        default=DEFAULT_OPENXR_VIEW_WIDTH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--view-height",
        type=int,
        default=DEFAULT_OPENXR_VIEW_HEIGHT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=DEFAULT_OPENXR_REMOTE_PORT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state_path = Path(args.state).expanduser() if args.state else None
    config_home = Path(args.config_home).expanduser() if args.config_home else None

    try:
        if args.command == "status":
            _print_state(status(state_path))
            return 0
        if args.command == "uninstall":
            current = status(state_path)
            if not current.installed:
                print("OpenVR Cardboard integration is not installed.")
                return 0
            uninstall(state_path=state_path, force=args.force)
            print("Removed the SteamVR driver and restored the OpenXR runtime.")
            return 0

        driver_source: Optional[Path] = None
        install_driver = not args.no_driver
        if install_driver:
            driver_source = (
                Path(args.driver).expanduser()
                if args.driver
                else _default_driver_source()
            )
            if driver_source is None:
                raise IntegrationError(
                    "no bundled driver was found; pass --driver PATH or --no-driver"
                )
            if find_steamvr(args.steamvr) is None:
                if args.steamvr:
                    raise IntegrationError(
                        f"SteamVR was not found at {args.steamvr}"
                    )
                # A machine without SteamVR is a supported configuration: the
                # phone HMD and the OpenXR demo still work. Only the explicit
                # --steamvr path is an error, because the user asked for it.
                print(
                    "SteamVR was not found; skipping the SteamVR driver. "
                    "OpenXR and phone streaming are unaffected.",
                    file=sys.stderr,
                )
                install_driver = False
                driver_source = None

        openxr_prefix: Optional[Path] = None
        register_openxr = not args.no_openxr
        if register_openxr:
            openxr_prefix = (
                Path(args.openxr).expanduser()
                if args.openxr
                else _default_openxr_prefix()
            )
            if openxr_prefix is None:
                if args.openxr:
                    raise IntegrationError(
                        f"OpenXR runtime folder does not exist: {args.openxr}"
                    )
                print(
                    "No bundled OpenXR runtime was found; skipping OpenXR "
                    "registration.",
                    file=sys.stderr,
                )
                register_openxr = False

        state = install(
            driver_source=driver_source,
            steamvr=args.steamvr,
            openxr_prefix=openxr_prefix,
            register_openxr=register_openxr,
            install_driver=install_driver,
            force=args.force,
            state_path=state_path,
            config_home=config_home,
            view_width=args.view_width,
            view_height=args.view_height,
            remote_port=args.remote_port,
        )
    except IntegrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if state.driver:
        print(f"Installed the SteamVR driver in {state.driver}")
        if state.driver_backup:
            print(f"Previous driver backed up to {state.driver_backup}")
    if state.openxr_active_runtime:
        print(f"Registered the OpenXR runtime at {state.openxr_active_runtime}")
    if state.driver or state.openxr_active_runtime:
        print("Restart SteamVR and any OpenXR application to use the new setup.")
        print("Run 'openvrcardboard-install uninstall' to undo these changes.")
    else:
        print("No integration changes were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
