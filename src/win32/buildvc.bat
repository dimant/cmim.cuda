@echo off
rem setlocal

call win32\setenv.bat

start /wait "vc" VCExpress.exe %1 /build "Release"

rem call "%VS100COMNTOOLS%\vsvars32.bat"

rem pushd ..\%prj%
rem start /wait "vc" VCExpress.exe %prj%.sln /build "Debug"
rem start /wait "vc" VCExpress.exe %prj%.sln /build "Release"
rem popd

rem set conf=Release
rem if "%~1" == "d" set conf=Debug

rem xcopy /y /d "..\%prj%\%conf%\*.dll" bin\debug
rem xcopy /y /d "..\%prj%\%conf%\*.dll" bin\release
rem xcopy /y /d "..\%prj%\%conf%\*.pdb" bin\debug
rem xcopy /y /d "..\%prj%\%conf%\*.pdb" bin\release

endlocal

