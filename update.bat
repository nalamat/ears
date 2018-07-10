@echo off

(goto 2>nul & ^
git fetch -t && ^
git checkout master -f && ^
git pull && ^
echo: && echo Update successful ... && pause >nul) || (
echo: && echo Update failed ... && pause >nul)
