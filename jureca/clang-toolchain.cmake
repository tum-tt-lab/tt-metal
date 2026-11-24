set(CMAKE_SYSTEM_PROCESSOR "x86_64")
# System clang (21.x on Rocky 10 eval-ttbh nodes); no lld/mold installed -> default GNU ld
set(CMAKE_C_COMPILER /usr/bin/clang CACHE INTERNAL "C compiler")
set(CMAKE_CXX_COMPILER /usr/bin/clang++ CACHE INTERNAL "C++ compiler")
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")
