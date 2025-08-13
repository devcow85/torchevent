#include "utils.h"

bool dbg_print_flag = true;

void dbg_print(const char *format, ...)
{
    if (!dbg_print_flag)
        return;

    va_list args;
    va_start(args, format);
    vprintf(format, args); // printf와 동일하게 처리
    va_end(args);
}