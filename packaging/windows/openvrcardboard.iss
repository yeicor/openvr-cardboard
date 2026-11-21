; Inno Setup script for the Windows desktop build.
;
; Built by scripts/build_windows_installer.ps1 after scripts/package.py has
; produced dist\openvrcardboard\. The installer copies that portable folder,
; runs the bundled openvrcardboard-install.exe to register the SteamVR driver,
; and unregisters it again on uninstall.
;
; ISCC /DAppVersion=1.2.3 /DSourceDir=C:\path\to\dist\openvrcardboard build.iss

#ifndef AppVersion
  #define AppVersion "1.2.1"
#endif

#ifndef SourceDir
  #define SourceDir "..\..\dist\openvrcardboard"
#endif

#define AppName "OpenVR Cardboard"
#define AppPublisher "OpenVR Cardboard contributors"

[Setup]
AppId={{63650591-EE4A-4FAB-8F3F-B19C65D17C40}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\OpenVR Cardboard
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename=OpenVR-Cardboard-{#AppVersion}-Setup
SetupIconFile=openvrcardboard.ico
UninstallDisplayIcon={app}\openvrcardboard.exe
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
CloseApplications=yes
RestartApplications=no
DisableDirPage=no
AllowNoIcons=yes
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "steamvr"; Description: "Install the SteamVR driver into the detected SteamVR installation"; GroupDescription: "SteamVR integration"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OpenVR Cardboard"; Filename: "{app}\openvrcardboard.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall OpenVR Cardboard"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OpenVR Cardboard"; Filename: "{app}\openvrcardboard.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Register the SteamVR driver. The bundled installer is reversible and backs
; up any previous folder; a missing SteamVR installation only prints a note
; and leaves the app usable with the Android/OpenXR paths.
Filename: "{app}\openvrcardboard-install.exe"; Parameters: "install --no-openxr"; WorkingDir: "{app}"; StatusMsg: "Installing the SteamVR driver..."; Flags: runhidden waituntilterminated; Tasks: steamvr
Filename: "{app}\openvrcardboard.exe"; Description: "{cm:LaunchProgram,OpenVR Cardboard}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\openvrcardboard-install.exe"; Parameters: "uninstall --no-openxr"; WorkingDir: "{app}"; RunOnceId: "OpenVRCardboardUninstallDriver"; Flags: runhidden waituntilterminated

[UninstallDelete]
Type: filesandordirs; Name: "{app}\__pycache__"
