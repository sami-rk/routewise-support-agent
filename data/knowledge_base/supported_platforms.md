# Supported platforms

## Stable clients

| Platform | Minimum version |
|---|---|
| Windows | Windows 10 and later |
| macOS | macOS 12 (Monterey) and later |
| iOS | iOS 15 and later |
| Android | Android 10 and later |

A device below the minimum version can still sign in to download your files, but
it cannot sync.

## Linux

The Linux client is in **beta**. It syncs and shares correctly, and it is not
covered by support for data loss. If you hit a problem on Linux, say so in your
ticket so we can prioritise it, and keep a local copy of anything you cannot
afford to lose.

## Web dashboard

The web dashboard at cloudsync.example.com works in any current browser and is
the quickest way to manage sharing links, two-factor authentication, and plan
changes. It does not upload local files; that needs a desktop or mobile client.

## File systems

Local folders sync on the drive and file system your operating system provides.
Network drives and filesystems that do not report changes reliably are not
supported.

## Common questions

**Do I need the desktop app to upload files?**
Yes. Use the desktop or mobile app to add files; the web dashboard manages your
account but does not upload local files.

**Does the Android client support offline mode?**
Yes. Offline mode works the same way on every stable client.

**Can I run the desktop app on a Linux server?**
The Linux client is in beta and is not intended for headless servers.
