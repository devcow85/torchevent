#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <unordered_set>

#include "transform.h"
#include "variables.h"
#include "core.h"
#include "utils.h"

namespace py = pybind11;

int test_func()
{
    return 42;
}

py::list SampleDPUTransform(py::array events)
{
    const int num_arrays = 3;
    const int array_size = 5;

    // 배열을 담을 Python 리스트
    py::list py_arrays;

    // 정적 배열 여러 개 생성 (예시)
    uint8_t arr_data[num_arrays][array_size] = {
        {1, 2, 3, 4, 5},
        {10, 11, 12, 13, 14},
        {100, 110, 120, 130, 140}};

    for (int i = 0; i < num_arrays; ++i)
    {
        py::list py_arr;
        for (int j = 0; j < array_size; ++j)
        {
            py_arr.append(arr_data[i][j]);
        }
        py_arrays.append(py_arr);
    }

    return py_arrays;
}

py::array_t<uint8_t> SimpleToFrame(py::array_t<NumpyEvent> events, py::tuple sensor_size)
{
    int sensor_width = sensor_size[0].cast<int>();
    int sensor_height = sensor_size[1].cast<int>();

    auto event_buffer_info = events.request();
    NumpyEvent *p = static_cast<NumpyEvent *>(event_buffer_info.ptr);

    py::array_t<short> temp({sensor_height, sensor_width});
    auto temp_buffer_info = temp.request();
    short *temp_ptr = static_cast<short *>(temp_buffer_info.ptr);

    for (auto i = 0; i < temp_buffer_info.size; i++)
    {
        temp_ptr[i] = 0;
    }

    for (int i = 0; i < event_buffer_info.size; i++)
    {
        NumpyEvent event = p[i];
        int x = event.x;
        int y = event.y;

        if (x >= 0 && x < sensor_width && y >= 0 && y < sensor_height)
        {
            temp_ptr[y * sensor_width + x] += 1;
        }
    }

    // Create a new numpy array to hold the result
    py::array_t<uint8_t> result_array(sensor_width * sensor_height);
    auto result_buffer_info = result_array.request();
    uint8_t *result_ptr = static_cast<uint8_t *>(result_buffer_info.ptr);

    // Fill the result array with the values from the temp array
    for (int i = 0; i < sensor_width * sensor_height; i++)
    {
        result_ptr[i] = static_cast<uint8_t>(temp_ptr[i]);
    }

    // Return the result array
    return result_array;
}

void DENOISE(DPU_INTERFACE *interface, DPU_COMMON *cm, uint8_t *event_frame_crop_buf)
{
    if (!interface->acc_map_on)
        return;

    DPU_PIC *acc = &cm->pic[PIC_TYPE_ACC];
    DPU_PIC *prv = &cm->pic[PIC_TYPE_ACC_PREV];
    DPU_PIC *evt = &cm->pic[PIC_TYPE_EVT];

    const int32_t eps_idx = interface->eps_idx;
    const int32_t *lut = argo_vars::lookuptable + eps_idx * 256 * 256;
    const int32_t pic_w = acc->width;
    const int32_t pic_h = acc->height;
    const int32_t stride = acc->stride;

#pragma omp parallel for collapse(2)
    for (int yy = 0; yy < pic_h; yy++)
    {
        for (int xx = 0; xx < pic_w; xx++)
        {
            int32_t offset = yy * stride + xx;

            uint8_t inp_val = acc->buf[offset];
            uint8_t out_val = prv->buf[offset];

            int32_t meanI = average_int(interface, acc->buf + offset, stride);
            int32_t meanII = square_average_int(interface, acc->buf + offset, stride);

            int32_t cov = meanII - meanI * meanI;
            int32_t a = lut[cov + 255];
            int32_t b = (meanI << 16) - a * meanI;

            int32_t y = (a * inp_val + b + (1 << 15)) >> 16;

            // Quantize to 16 levels (step of 16)
            y = (y + 8) & ~15;

            int32_t result = CLIP3(128 + (int)out_val - y);
            event_frame_crop_buf[yy * pic_w + xx] = result;
            prv->buf[offset] = CLIP3(y);
        }
    }
}

struct PointHash
{
    std::size_t operator()(const std::pair<int, int> &p) const
    {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

bool skip_hotpixels(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    // static const std::unordered_set<std::pair<int, int>, PointHash> skip_set = {
    //     {945, 561}, {1213, 687}, {43, 610}, {511, 118}, {563, 204}, {1087, 681}, {311, 325}, {1023, 313}, {746, 659}, {383, 175}, {480, 440}, {272, 85}};

    // int x = cm->dpu_event.X;
    // int y = cm->dpu_event.Y;

    // return skip_set.find({x, y}) != skip_set.end();
    return false; //! KETI의 EVK4 카메라 사용중일때만 위의 코드 사용!
}

py::array_t<uint8_t> argo_toframe(py::array events, py::dict kwargs)
{
    int width = kwargs["width"].cast<int>();
    int height = kwargs["height"].cast<int>();
    int linear_map_log2scale = kwargs["linear_map_log2scale"].cast<int>();
    int eps_idx = kwargs["eps_idx"].cast<int>();
    int filter_radius = kwargs["filter_radius"].cast<int>();

    DPU_INTERFACE interface;
    DPU_COMMON common;

    DPU_INTERFACE *it = &interface;
    DPU_COMMON *cm = &common;

    init_dpu(it, cm, width, height, linear_map_log2scale, eps_idx, filter_radius);

    int ret = 0;
    int j = 0;
    int count = 0;

    DPU_EVENT *store_buf = cm->dpu_crop.dpu_store_buf;

    int count_total = 0;
    int num_event_per_packet = 0;

    cm->prev_ts = cm->dpu_event.timestamp;

    auto event_buffer_info = events.request();
    NumpyEvent *p_events = static_cast<NumpyEvent *>(event_buffer_info.ptr);

    // calc num_frame
    NumpyEvent last_event = p_events[event_buffer_info.size - 1];
    int last_time = last_event.t;
    int last_time_count = last_time / 1111;
    int num_frame = last_time_count / 6;

    py::array_t<uint8_t> result_frames = py::array_t<uint8_t>({num_frame, cm->pic[PIC_TYPE_EVT].height, cm->pic[PIC_TYPE_EVT].width});
    py::buffer_info buf_info = result_frames.request();
    uint8_t *base_ptr = static_cast<uint8_t *>(buf_info.ptr);

    for (size_t i = 0; i < event_buffer_info.size; i++)
    {
        NumpyEvent event = p_events[i];

        // Converting NumpyEvent to DPU_EVENT
        cm->dpu_event.timestamp = event.t / 1111;
        cm->dpu_event.X = event.x & 0x0FFF;
        cm->dpu_event.Y = event.y & 0x07FF;
        cm->dpu_event.polarity = event.p & 0x0001;

        if (skip_hotpixels(it, cm))
        {
            continue;
        }

        cm->eventnumcount++;

        if (cm->prev_ts != cm->dpu_event.timestamp)
        {
            count++;
            if (count == 6)
            {
                // add frame offset
                int offset = cm->frame_count * cm->pic[PIC_TYPE_EVT].height * cm->pic[PIC_TYPE_EVT].width;
                auto temp_ptr = base_ptr + offset;

                DENOISE(it, cm, temp_ptr);

                memreset_eventcount(it, cm);
                memreset_event(it, cm);
                memreset_eventframecrop(it, cm);

                store_buf = cm->dpu_crop.dpu_store_buf;

                cm->frame_count++;
                cm->prv_evt_count = cm->evt_count;
                cm->pre_time_stamp = cm->dpu_event.timestamp;
                cm->eventnumcount = 0;
                cm->windowcount = 0;
                count = 0;
            }
        }
        ACC_MAP(it, cm);
        cm->evt_count++;
        cm->prev_ts = cm->dpu_event.timestamp;
        j++;
    }

    finish_dpu(it, cm);
    return result_frames;
}

PYBIND11_MODULE(libargo, m)
{
    PYBIND11_NUMPY_DTYPE(NumpyEvent, x, y, p, t);

    m.doc() = "pybind11 plugin for libargo"; // optional module docstring
    m.def("test_func", &test_func, "A function that returns 42");
    m.def("sample_tranform", &SampleDPUTransform, "A function that returns a list of arrays");
    m.def("simple_toframe", &SimpleToFrame, "A function that transforms events to frame");
    m.def("argo_toframe", &argo_toframe, "A function that transforms events to frame");
}