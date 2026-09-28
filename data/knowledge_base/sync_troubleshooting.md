# Sync troubleshooting

Work down the list. Each step rules out a common cause, and the first three fix
most problems on their own.

## 1. Check the status marker

Look at the sync folder. A tick means synced, a circle means work in progress, an
exclamation mark means that file failed. Note which files have the exclamation
mark before you change anything.

## 2. Check you are online

Offline mode keeps editing available, so you can make changes that have nowhere
to go. Look for the offline indicator in the app's status bar.

## 3. Check your device allowance

**Account > Subscription** shows devices used against devices allowed. Basic
allows 2, Pro allows 5, Business is unlimited. If you are over, the newest
device signs in but does not sync. Sign out on a device you no longer use.

## 4. Check storage

Uploads pause when your account storage is full. If uploads stopped and you
recently added large files, free space or upgrade your plan. Download your files
and delete them from the device if the device's own storage is full.

## 5. Check the file size

Basic caps a single file at 2 GB. Pro and Business cap it at 10 GB. A larger file
fails with an error marker and does not retry.

## 6. Restart the app

Quit the desktop app completely, including the system tray icon, and start it
again. This clears a stuck upload or download queue.

## 7. Files look different on two devices

That is usually a conflict rather than data loss. Both versions are kept for 30
days. Open **version history** on the file and restore the version you want,
then sync.

## 8. Sync is stuck at 99%

A very large file or many small changes at once can leave the progress bar
sitting at 99 while the last part uploads. Leave it running; if it has not moved
in an hour, restart the app as in step 6.

## When to open a ticket

Open a ticket if the app still fails after the steps above, if one device shows
an error the others do not, or if files are missing that you cannot find in
version history. Include your platform, your app version, the name of the file,
and the time the problem started.

## Common questions

**Is the Linux client supported?**
It is in beta. See supported platforms.

**Why did sync stop on my phone but work on my laptop?**
The phone is most likely over the device allowance for your plan. Check
**Account > Subscription**.

**My files disappeared after an update.**
Check version history first; 30 days of versions are kept. If the file is not
there, open a ticket with your device details and we will investigate as a
possible data loss issue.
