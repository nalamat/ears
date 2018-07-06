@echo off

(goto 2>nul & git fetch -t && ^
git checkout RC -f && ^
echo: && echo Update successful ...) || ^
(echo: && echo Update failed ... && pause >nul)
