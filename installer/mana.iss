; installer/mana.iss -- Inno Setup script for MANA.
;
; Built by scripts/build_installer.py, which fills in MyAppVersion and
; checks that dist\MANA actually exists first. Compiling this file by hand
; works too; the script only adds the checks.
;
; ----------------------------------------------------------------------
; Why this installs into LocalAppData and not Program Files
; ----------------------------------------------------------------------
; MANA rewrites its own source. `code_evolution.apply_patch` opens a file
; under the install directory, writes a new version of it and records the
; change in a changelog that later runs read back. Program Files is not
; writable by a standard user, so an installation there would leave the
; application's single defining capability failing at runtime with a
; permission error -- and failing quietly, because the changelog write
; would fail too.
;
; So: {localappdata}\Programs\MANA, PrivilegesRequired=lowest. No UAC
; prompt, no elevation, and the agent can patch itself as the user who
; runs it. This is the same choice VS Code and several other
; self-updating applications make, for the same reason.
;
; ----------------------------------------------------------------------
; What is NOT removed on uninstall
; ----------------------------------------------------------------------
; %LOCALAPPDATA%\MANA holds the knowledge database, the state pickle, the
; cognitive genome and the changelog -- everything the agent has learned.
; paths.py places it outside the install directory precisely so it
; survives reinstalls, and the uninstaller leaves it alone. Deleting an
; agent's memory because its program was uninstalled is a data loss the
; user did not ask for; the [UninstallDelete] section below removes only
; what the installer itself put there.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\MANA"
#endif

[Setup]
AppId={{8E5C1A94-3D6F-4B2E-9A17-2C4B7F0E5D31}
AppName=MANA
AppVersion={#MyAppVersion}
AppVerName=MANA {#MyAppVersion}
AppPublisher=Aleksey Burtsev
DefaultDirName={localappdata}\Programs\MANA
DefaultGroupName=MANA
DisableProgramGroupPage=yes
; No elevation: see the header. An installer that asked for admin here
; would produce an application that cannot modify its own code.
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=MANA-{#MyAppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; "x64compatible" requires Inno Setup 6.3 or newer.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=MANA {#MyAppVersion}
UninstallDisplayIcon={app}\MANA.exe

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Запускать MANA при входе в систему"; GroupDescription: "Дополнительно:"; Flags: unchecked

[Files]
; Everything PyInstaller produced, recursively: the executable, the
; Python runtime, mana\ as loose .py files (patchable -- that is the
; point), python\ as the sandbox interpreter, mana_desktop\web\ and
; build_manifest.json, which records what this particular build contains.
; ignoreversion overwrites unconditionally, which on an upgrade also
; replaces any mana\*.py the agent rewrote after installation. That is
; the correct behaviour -- a new version is a new program -- but it
; means the changelog in %LOCALAPPDATA%\MANA will refer to patches
; that are no longer in the code. The learned state survives; the
; self-applied source edits do not.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MANA"; Filename: "{app}\MANA.exe"
Name: "{group}\Диагностика MANA"; Filename: "{app}\MANA.exe"; Parameters: "--self-check"; Comment: "Проверить, что сборка сохранила песочницу, самопатчинг и зависимости"
Name: "{autodesktop}\MANA"; Filename: "{app}\MANA.exe"; Tasks: desktopicon

[Registry]
; HKCU, matching a per-user install. An autostart entry in HKLM would
; outlive an uninstall run by a different user.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "MANA"; ValueData: """{app}\MANA.exe"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\MANA.exe"; Description: "{cm:LaunchProgram,MANA}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; __pycache__ and any .py the agent rewrote after installation: files
; that appeared inside {app} after setup ran, which Inno does not track
; and would otherwise leave behind as an orphaned directory.
Type: filesandordirs; Name: "{app}\mana\__pycache__"
Type: filesandordirs; Name: "{app}\mana_desktop\__pycache__"
Type: dirifempty; Name: "{app}"

[Messages]
russian.WelcomeLabel2=Будет установлена [name/ver].%n%nMANA устанавливается в папку пользователя и не требует прав администратора: приложение изменяет собственный код, а для этого каталог установки должен быть доступен ему на запись.%n%nПамять агента при удалении сохраняется — она лежит отдельно от программы.
