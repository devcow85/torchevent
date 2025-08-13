#include <math.h>
#include <stdbool.h>
#include <vector>

#include "transform.h"

// DVC Format
// #pragma pack(push, 1)
// struct NumpyEvent
// {
//     uint32_t t;
//     uint16_t x;
//     uint16_t y;
//     uint8_t p;
// };
// // using 9 bytes packing
// #pragma pack(pop)

// for evk4 : dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16})
#pragma pack(push, 1)
typedef struct NumpyEvent
{
    uint16_t x; // 2
    uint16_t y; // 2
    int16_t p;  // 2
    // uint16_t pad; // 2 bytes padding (to match NumPy)
    uint64_t t; // 8
} NumpyEvent;
#pragma pack(pop)

void init_interface(DPU_INTERFACE *interface, int width, int height, int linear_map_log2scale, int eps_idx, int filter_radius);
const char *get_dpu_savepath(DPU_COMMON *cm, const char *folder, MODULES module);
void init_dpu(DPU_INTERFACE *interface, DPU_COMMON *common, int width, int height, int linear_map_log2scale, int eps_idx, int filter_radius);
void finish_dpu(DPU_INTERFACE *interface, DPU_COMMON *cm);