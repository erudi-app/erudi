# NSIS include for the Windows installer (electron-builder `nsis.include`).
#
# Installing over an existing Erudi (a manual install or an auto-update) runs the
# uninstaller of the INSTALLED version with --updated. That uninstaller
# (electron-builder's un.atomicRMDir) moves every file of the install into
# %TEMP%\nsXXXX.tmp\old-install\: it recreates each directory there with
# CreateDirectory and moves each file with NSIS `Rename` (MoveFileW), both
# without the \\?\ prefix, so a directory of 248 characters or more, or a file of
# 260 (MAX_PATH) or more, cannot be written. On the first file it cannot move it
# moves back what it had moved and exits with code 2, leaving the old install
# intact.
#
# The deepest files of the install are license notices that wheels ship under
# <name>.dist-info\licenses\ (torch carries its vendored third_party tree a
# dozen directories deep): LICENSE.txt files 200 characters below the install
# directory, in directories 188 characters below it. Under
#   C:\Users\<user>\AppData\Local\Temp\nsXXXX.tmp\old-install
# (48 characters plus the user name plus the 1 to 4 hex digits of the plugins
# directory name, nsX + hex + .tmp) both limits are reached at the same length,
# so the failure starts at a user name of 8 to 11 characters, depending on that
# random name: possible from 8, certain from 11.
#
# Two macros remove the dist-info license folders of the OLD install's backend
# so its uninstaller never has to move them. Nothing else is touched, and the new
# install ships the same notices again.
#
# - customInit (end of .onInit) removes them for every install over an existing
#   one, an update (--updated, how electron-updater starts the installer) or an
#   installer started by hand. The old uninstaller's first attempt then
#   succeeds, and electron-builder never reaches its retry loop (five attempts)
#   or its "cannot be closed... click Retry" dialog, whose Retry repeats the
#   failing uninstall. The cost: .onInit runs before the installer's pages, so
#   a user who cancels an interactive install after it keeps the old install,
#   working, without its license notices until the next install.
# - customUnInstallCheck / customUnInstallCheckCurrentUser are the safety net
#   for an old install that customInit did not clean (an unrecognised location,
#   a folder it could not delete). electron-builder's handleUninstallResult
#   (installUtil.nsh) inserts them IN PLACE OF its own check of the old
#   uninstaller's exit code, so they reproduce that check: an uninstaller that
#   cannot be launched is logged and the install goes on; an uninstaller that
#   exits non-zero shows $(uninstallFailed), sets exit code 2 and quits. Before
#   giving up on a non-zero exit, they remove the folders and run the old
#   uninstaller once more. A successful uninstall (exit code 0) takes none of
#   this code.

!include "LogicLib.nsh"

# Install directory of the old version: InstallLocation under
# INSTALL_REGISTRY_KEY, written by every electron-builder installer and read the
# same way by uninstallOldVersion. Neither its global $installationDir nor its
# readReg macro can be used here: installUtil.nsh declares them after .onInit
# and after handleUninstallResult, where these macros expand (an unknown
# variable is a warning, and electron-builder builds with -WX).
!macro erudiRemoveOldBackendLicenses ROOT_KEY
  Push $R7
  Push $R8
  Push $R9

  ReadRegStr $R9 ${ROOT_KEY} "${INSTALL_REGISTRY_KEY}" InstallLocation

  # Guard: an absolute drive path ("C:\..."), longer than a drive root, that
  # holds this app's executable and a frozen backend. Anything else (no
  # install, "C:\", a UNC share, a relative path) removes nothing.
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
  ${ElseIf} $R9 != ""
    DetailPrint `Old install location "$R9" not recognised; no file removed.`
  ${EndIf}

  Pop $R9
  Pop $R8
  Pop $R7
  # ReadRegStr, FindFirst and FindNext set the error flag on a missing value or
  # the end of the listing; the caller's next test of the flag must not see it.
  ClearErrors
!macroend

!macro customInit
  # initMultiUser has just pointed SHELL_CONTEXT at the existing install's
  # hive; a per-machine install also replaces a per-user one, as the install
  # section does. With no existing install the registry value is empty and
  # nothing is removed.
  !insertmacro erudiRemoveOldBackendLicenses SHELL_CONTEXT
  ${If} $installMode == "all"
    !insertmacro erudiRemoveOldBackendLicenses HKEY_CURRENT_USER
  ${EndIf}
!macroend

!macro erudiHandleUninstallResult ROOT_KEY
  ${If} ${Errors}
    DetailPrint `Uninstall was not successful. Not able to launch uninstaller!`
  ${ElseIf} $R0 != 0
    DetailPrint `Uninstaller error code: $R0. Retrying once without the old backend's license folders.`
    !insertmacro erudiRemoveOldBackendLicenses ${ROOT_KEY}
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
