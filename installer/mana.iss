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

[Code]
// WebView2 draws MANA's window. Windows 11 ships it; Windows 10 does not
// guarantee it. Saying so during setup is worth a great deal more than
// letting the first double-click do nothing at all -- and setup is the
// moment the person is already prepared to install something.
//
// A warning, not a refusal: MANA runs from the command line without a
// window, and the runtime can be installed afterwards.
function WebView2Installed(): Boolean;
var
  Guid, Version: String;
begin
  // The braces are built with Chr rather than written literally, and that
  // is not fussiness. Written as '{{...}}' the path came out with doubled
  // braces, the registry lookup missed, and the warning would have fired
  // on every machine INCLUDING ones that have WebView2 -- verified by
  // logging the result on a machine with 152.0.4191.66 installed, which
  // reported WebView2Installed=0. The compiler accepts both spellings, so
  // only running it catches this.
  Guid := Chr(123) + 'F3017226-FE2A-4295-8BDF-00C3A9A7E4C5' + Chr(125);
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + Guid, 'pv', Version) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + Guid, 'pv', Version) or
    RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + Guid, 'pv', Version);
  if Result then
    Result := (Version <> '') and (Version <> '0.0.0.0');
  Log('MANA: WebView2Installed=' + IntToStr(Integer(Result)) + ' pv=' + Version);
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not WebView2Installed() then
  begin
    MsgBox('На этом компьютере не найден WebView2 Runtime.'#13#10#13#10 +
           'Им рисуется окно MANA. В Windows 11 он встроен, в Windows 10 —' +
           ' нет.'#13#10#13#10 +
           'Установка продолжится: MANA работает и из командной строки' +
           ' (MANA.exe --cli). Чтобы открывалось окно, поставьте' +
           ' «Evergreen Standalone Installer» отсюда:'#13#10 +
           'https://developer.microsoft.com/microsoft-edge/webview2/',
           mbInformation, MB_OK);
  end;
end;

[Messages]
russian.WelcomeLabel2=Будет установлена [name/ver].%n%nMANA устанавливается в папку пользователя и не требует прав администратора: приложение изменяет собственный код, а для этого каталог установки должен быть доступен ему на запись.%n%nПамять агента при удалении сохраняется — она лежит отдельно от программы.
