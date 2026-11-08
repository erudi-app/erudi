# NSIS include for the Windows installer (electron-builder `nsis.include`).
#
# Installing over an existing Erudi (a manual install or an auto-update) runs the
# uninstaller of the INSTALLED version with --updated. That uninstaller
# (electron-builder's un.atomicRMDir) moves every file of the install into
# %TEMP%\nsXXXX.tmp\old-install\ with NSIS `Rename`, a MoveFileW call without
# the \\?\ prefix, so it fails on any destination of MAX_PATH (260) characters
# or more. It then moves back what it had moved and exits with code 2, leaving
# the old install intact. The deepest files of the install are license notices
# that wheels ship under <name>.dist-info\licenses\ (torch carries its vendored
# third_party tree a dozen directories deep): 200 characters below the install
# directory, so under C:\Users\<user>\AppData\Local\Temp\nsXXXX.tmp\old-install\
# they reach 260 for a user name of about 8 characters (the plugins directory
# name, nsX plus 1 to 4 hex digits plus .tmp, moves the limit by a character or
# two).
#
# electron-builder's handleUninstallResult (installUtil.nsh) inserts these two
# macros IN PLACE OF its own failure handling, so they reproduce it: an
# uninstaller that cannot be launched is logged and the install goes on; an
# uninstaller that exits non-zero shows $(uninstallFailed), sets exit code 2
# and quits. Before giving up on a non-zero exit, they make one recovery
# attempt: delete the dist-info license folders of the OLD install's backend
# and run the old uninstaller once more. The old install is being replaced, the
# new one ships the same notices again, and nothing else is touched. A
# successful uninstall (exit code 0) takes none of this code.

!include "LogicLib.nsh"

# Install directory of the old version, read the way uninstallOldVersion reads
# it (InstallLocation under INSTALL_REGISTRY_KEY, written by every
# electron-builder installer). uninstallOldVersion keeps it in the global
# $installationDir, but that variable is declared further down installUtil.nsh
# than handleUninstallResult, where these macros expand, so it cannot be named
# here (an unknown variable is a warning, and electron-builder builds with -WX).
!macro erudiRemoveOldBackendLicenses ROOT_KEY
  Push $R7
  Push $R8
  Push $R9

  !insertmacro readReg $R9 "${ROOT_KEY}" "${INSTALL_REGISTRY_KEY}" InstallLocation

  # Guard: an absolute drive path ("C:\..."), longer than a drive root, that
  # holds this app's executable and a frozen backend. Anything else (empty,
  # "C:\", a UNC share, a relative path) skips the recovery and falls through
  # to the default failure handling.
  StrCpy $R8 $R9 1 -1
  ${If} $R8 == "\"
    StrCpy $R9 $R9 -1
  ${EndIf}
  StrLen $R8 $R9
  StrCpy $R7 $R9 2 1
  ${If} $R8 > 3
  ${AndIf} $R7 == ":\"
  ${AndIf} ${FileExists} "$R9\${APP_EXECUTABLE_FILENAME}"
  ${AndIf} ${FileExists} "$R9\resources\backend\_internal\*.*"
    # \\?\ hands the path to the W file APIs unparsed, which lifts the MAX_PATH
    # limit (Unicode NSIS calls FindFirstFileW / DeleteFileW / RemoveDirectoryW,
    # and RMDir /r builds each child path by appending to the one it was given).
    # In place the deepest notice sits at 238 characters plus the user name, so
    # without the prefix this deletion would fail for a 22-character user name
    # or a deeper custom install directory. The prefix also disables path
    # normalisation: a "/" or a ".." in the registry value makes the path
    # resolve to nothing rather than to another directory.
    StrCpy $R9 "\\?\$R9\resources\backend\_internal"
    DetailPrint `Removing the dist-info license folders of the old backend in $R9`
    ClearErrors
    FindFirst $R7 $R8 "$R9\*.dist-info"
    ${DoWhile} $R8 != ""
      ${If} ${FileExists} "$R9\$R8\licenses\*.*"
        RMDir /r "$R9\$R8\licenses"
        ${If} ${FileExists} "$R9\$R8\licenses\*.*"
          DetailPrint `Could not remove $R9\$R8\licenses`
        ${EndIf}
      ${EndIf}
      FindNext $R7 $R8
    ${Loop}
    FindClose $R7
  ${Else}
    DetailPrint `Old install location "$R9" not recognised; no file removed before the second attempt.`
  ${EndIf}

  Pop $R9
  Pop $R8
  Pop $R7
!macroend

!macro erudiHandleUninstallResult ROOT_KEY
  ${If} ${Errors}
    DetailPrint `Uninstall was not successful. Not able to launch uninstaller!`
  ${ElseIf} $R0 != 0
    DetailPrint `Uninstaller error code: $R0. Retrying once without the old backend's license folders.`
    !insertmacro erudiRemoveOldBackendLicenses "${ROOT_KEY}"
    # `Call`, not `!insertmacro uninstallOldVersion`: that macro is defined
    # after handleUninstallResult in installUtil.nsh, and a Call resolves at
    # link time. The function clears the error flag and sets $R0 itself.
    Push "${ROOT_KEY}"
    Call uninstallOldVersion
    ${If} ${Errors}
      DetailPrint `Uninstall was not successful. Not able to launch uninstaller!`
    ${ElseIf} $R0 != 0
      MessageBox MB_OK|MB_ICONEXCLAMATION "$(uninstallFailed): $R0"
      DetailPrint `Uninstall was not successful. Uninstaller error code: $R0.`
      SetErrorLevel 2
      Quit
    ${EndIf}
  ${EndIf}
!macroend

!macro customUnInstallCheck
  !insertmacro erudiHandleUninstallResult SHELL_CONTEXT
!macroend

!macro customUnInstallCheckCurrentUser
  !insertmacro erudiHandleUninstallResult HKEY_CURRENT_USER
!macroend
