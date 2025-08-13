#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>

#if (defined(_WIN32) || defined(__WIN32__))
#include <direct.h>
#define mkdir(x) _mkdir(x)
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdbool.h>

#define LOG_E(x) log(x)
#define PADDING 16
#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT)
#define FRAME_BITDEPTH 8
#define MIN_VAL 0
// #define MAX_VAL (1<<FRAME_BITDEPTH)-1
#define MAX_VAL (1 << FRAME_BITDEPTH) - 16

#define OUTPUT_FRAME_WIDTH 1280
#define OUTPUT_FRAME_HEIGHT 720
#define DPU_STORE_SIZE 50000

#define BUFFER_NUM PIC_TYPE_ALL
#define BUFFER_WIDTH (FRAME_WIDTH + (PADDING << 1))
#define BUFFER_HEIGHT (FRAME_HEIGHT + (PADDING << 1))
#define BUFFER_SIZE BUFFER_WIDTH *BUFFER_HEIGHT

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLIP3(x) MIN(MAX(x, MIN_VAL), MAX_VAL)

#ifndef KW_COMMON_H_
#define KW_COMMON_H_

typedef enum MODULE_TYPES
{
    MODULE_ACC_MAP,
    MODULE_HDEVENTFRAME,
    MODULE_ROIEVENTFRAME,
    MODULE_RESIZEEVENTFRAME,
    MODULE_FRAME_HDSPIKE,
    MODULE_FRAME_ROISPIKE,
    MODULE_FRAME_RESIZESPIKE,
    MODULE_HDSPIKE,
    MODULE_ROISPIKE,
    MODULE_RESIZESPIKE,
    MODULE_COMPRESS_EVENTFRAME,
    MODULE_COMPRESS_EVENT,
    MODULE_TXT,
    MODULE_ALL
} MODULES;

typedef enum PIC_TYPES
{
    PIC_TYPE_ACC,
    PIC_TYPE_ACC_PREV,
    PIC_TYPE_EVT,
    PIC_TYPE_R,
    PIC_TYPE_G,
    PIC_TYPE_B,
    PIC_TYPE_RGB_PREV,
    PIC_TYPE_ALL
} DPU_PIC_TYPE;

typedef enum BUF_TYPES
{
    BUF_TYPE_STORE,
    BUF_TYPE_CROP,
    BUF_TYPE_ALL,
} BUF_TYPE;

typedef uint64_t timestamp_t; // Type for timestamp, in microseconds

typedef enum
{
    CD,
    EM
} EvType;

typedef struct DAT_EVENT
{
    uint32_t timestamp : 32;
    uint32_t X : 14;
    uint32_t Y : 14;
    uint32_t polarity : 4;
} DAT_EVENT;

typedef struct DPU_EVENT
{
    uint32_t timestamp : 8; // [07:00] - 10 ms unit
    uint32_t X : 12;        // [19:08]
    uint32_t Y : 11;        // [30:20]
    uint32_t polarity : 1;  // [31:31]
} DPU_EVENT;

typedef struct DPU_PIC
{
    uint8_t *buf_origin;
    int32_t stride;
    uint8_t *buf;
    int32_t width;
    int32_t height;
} DPU_PIC;

typedef struct DPU_PIC_2
{
    uint16_t *buf_origin;
    int32_t stride;
    uint16_t *buf;
    int32_t width;
    int32_t height;
} DPU_PIC_2;

typedef struct DPU_INTERFACE
{
    // COMMONs
    uint16_t input_width : 16;
    uint16_t input_height : 16;
    uint16_t bitdepth : 4;
    // INPUT EVENT FORMAT
    uint16_t evt_format : 1; // 0 : DAT, 1 : EVT3.0
    // DAT2DVT
    uint16_t evt_cvt_on : 1;
    // ACC_MAP
    uint16_t acc_map_on : 1;
    uint16_t linear_map_log2scale : 4;
    // DENOISE
    uint16_t eps_idx : 2;
    uint16_t filter_radius : 1;
    // COMPRESS
    uint16_t eventframe_compress_on : 1;
    uint16_t event_compress_on : 1;
    // RESIZE
    uint16_t resize_on : 1;
    uint16_t input_type : 2;
    uint16_t downsampling_method : 2;
    // EVENTCOUNT
    uint16_t eventcount_on : 1;
    // ROI
    uint16_t roi_on : 1;
    // AUTO EXPOSURE
    uint16_t autoexposure_on : 1;
    uint16_t max_acc_time : 2;
    // EVENT FRAME CROP
    uint16_t eventframecrop_on : 1;
    // EVENT CROP
    uint16_t eventcrop_on : 1;
    // ROI
    uint16_t static_roi_flag : 1;
    // V2E
    uint16_t video2eventframe_on : 1;
    uint16_t v2e_threshold : 3;
    // EVENT FRAME TO SPIKE
    uint16_t ef2spike_on : 1;
    uint16_t spike_method : 1;
} DPU_INTERFACE;

typedef struct DPU_CROP_COMMON
{
    DPU_EVENT *dpu_store_buf;
    DPU_EVENT *dpu_crop_buf;
    uint16_t *count_buf;
    uint16_t *offset_buf;
} DPU_CROP_COMMON;

typedef struct HISTOGRAM
{
    uint16_t *histogram_x;
    uint16_t *histogram_y;
    uint32_t mean_x;
    uint32_t mean_y;
} HISTOGRAM;

typedef struct ROI
{
    uint16_t *roi_buf;
    uint8_t roi_count;
    uint16_t *static_roix;
    uint16_t static_roiy;
} ROI;

// Evt3 raw events are 16-bit words
typedef struct RawEvent
{
    uint16_t pad : 12; // Padding
    uint16_t type : 4; // Event type
} RawEvent;

typedef struct DPU_COMMON
{
    char wkdir[1000];            // 파일 쓸때
    char dirs[MODULE_ALL][1000]; // 파일 쓸때
    FILE *fp_dat;                // dat 읽을때
    FILE *fp_dpu;                // dpu 쓸때
    FILE *fp_accmap;             // accmap 쓸때
    FILE *fp_hdeventframe;       // hdeventframe 쓸때
    FILE *fp_rgb;                // rgb 읽을때
    FILE *fp_crop_eventframe;
    FILE *fp_crop_event;
    FILE *fp_framehd_dpu;
    uint8_t *pic_pool;        // hd 해상도 변수 선언시
    uint8_t *pic_resize_pool; // resized 해상도 변수 선언시
    uint16_t *pic_v2e_pool;   // resized 해상도 변수 선언시
    DPU_PIC pic[PIC_TYPE_ALL];
    DPU_PIC_2 pic2[PIC_TYPE_ALL];
    DAT_EVENT dat_event;
    DAT_EVENT *dat_event_12; //  maximum 12 events decoded from one EVT3.0 packet
    DPU_EVENT dpu_event;
    DPU_EVENT resize_event;
    DPU_EVENT ef2spike_event;
    // DPU_EVENT spike_event;
    DAT_EVENT *frame_event;      // 이벤트2스파이크 변환시
    uint16_t *frame_event_count; // 이벤트2스파이크 변환시
    RawEvent raw_event;          // evt3.0읽을때
    HISTOGRAM histogram;         // 이벤트 히스토그램을 위해 선언
    ROI roi;                     // roi정보 저장
    uint8_t auto_exposure_flag;  // 누적 반복 여부 결정하는 flag
    DPU_CROP_COMMON dpu_crop;    // 이벤트 크롭시
    DPU_CROP_COMMON frame_crop;  // 이벤트 프레임 크롭 시
    uint8_t *event_frame_crop_buf;
    uint32_t *event_frame_crop_buf_size;
    // uint8_t* event_frame_crop_buf_resize;
    // uint32_t* event_frame_crop_buf_resize_size;
    int windowcount;     // auto exposure에서 윈도우 누적 얼마나 됐는지 확인
    int frame_count;     // 프레임 수 printf를 위한거
    int frameeventcount; // 이벤트에서 스파이크 만들때 스파이크 개수 카운트
    int prev_ts;         // 프레임 처리를 위해 시간 비교 용
    int evt_count;       // 이벤트 개수 카운트
    int prv_evt_count;   // auto exposure에서 이벤트 개수 새기 위해서 사용
    int pre_time_stamp;  // auto exposure에서 flag결정을 위해 사용
    // compress
    char filename_bitstream_eventframe[1000];
    char filename_bitstream_event[1000];
    char filename_roi_event[1000];
    char filename_roi_eventframe[1000];
    char filename_resize_event[1000];
    char bitstream_save_path[1000];
    char filename_resize_eventframe[1000];
    char filename_frame_roi_event[1000];
    char filename_frame_resize_event[1000];
    char filename_roi_txt[1000];

    // read evt3.0
    timestamp_t current_time_base; // time high bits
    timestamp_t current_time_low;
    timestamp_t current_time;
    uint16_t current_cd_y;
    uint16_t current_x_base;
    uint16_t current_polarity;
    unsigned int n_time_high_loop; // Counter of the time high loops
    EvType current_type;
    bool first_time_base_set;

    int eventnumcount; // eventcount 모듈에서 사용
} DPU_COMMON;

// For EVT3.0 decoding
enum EventTypes
{
    CD_Y = 0x0,
    EM_Y = 0x1,
    X_POS = 0x2,
    X_BASE = 0x3,
    VECT_12 = 0x4,
    VECT_8 = 0x5,
    EVT_TIME_LOW = 0x6,
    EVT_TIME_HIGH = 0x8,
    EXT_TRIGGER = 0xA
};
// typedef uint8_t EventTypes;

typedef struct RawEventTime
{
    uint16_t time : 12;
    uint16_t type : 4; // Event type : EventTypes::EVT_TIME_LOW OR EventTypes::EVT_TIME_HIGH
} RawEventTime;

typedef struct RawEventXPos
{
    uint16_t x : 11;   // Pixel X coordinate
    uint16_t pol : 1;  // Event polarity:
                       // '0': decrease in illumination
                       // '1': increase in illumination
    uint16_t type : 4; // Event type : EventTypes::X_POS
} RawEventXPos;

typedef struct RawEventVect12
{
    uint16_t valid : 12; // Encodes the validity of the events in the vector :
                         // foreach i in 0 to 11
                         //   if valid[i] is '1'
                         //      valid event at X = X_BASE.x + i
    uint16_t type : 4;   // Event type : EventTypes::VECT_12
} RawEventVect12;

typedef struct RawEventVect8
{
    uint16_t valid : 8; // Encodes the validity of the events in the vector :
                        // foreach i in  0 to 7
                        //   if valid[i] is '1'
                        //      valid event at X = X_BASE.x + i
    uint16_t unused : 4;
    uint16_t type : 4; // Event type : EventTypes::VECT_8
} RawEventVect8;

typedef struct RawEventY
{
    uint16_t y : 11;   // Pixel Y coordinate
    uint16_t orig : 1; // Identifies the System Type:
                       // '0': Master Camera (Left Camera in Stereo Systems)
                       // '1': Slave Camera (Right Camera in Stereo Systems)
    uint16_t type : 4; // Event type : EventTypes::CD_Y OR EventTypes::EM_Y
} RawEventY;

typedef struct RawEventXBase
{
    uint16_t x : 11;   // Pixel X coordinate
    uint16_t pol : 1;  // Event polarity:
                       // '0': decrease in illumination
                       // '1': increase in illumination
    uint16_t type : 4; // Event type : EventTypes::X_BASE
} RawEventXBase;

typedef struct RawEventExtTrigger
{
    uint16_t value : 1; // Trigger current value (edge polarity):
                        // - '0' (falling edge);
                        // - '1' (rising edge).
    uint16_t unused : 7;
    uint16_t id : 4;   // Trigger channel ID.
    uint16_t type : 4; // Event type : EventTypes::EXT_TRIGGER
} RawEventExtTrigger;

static const char *module_ext[MODULE_ALL] = {
    "0_0_1280_720.raw",
    "0_0_1280_720.raw",
    ".roieventframe.raw",
    ".resizeeventframe.raw",
    "0_0_1280_720.dvt",
    ".frame_roispike.dvt",
    ".frame_resizespike.dvt",
    "0_0_1280_720.dvt",
    ".roispike.dvt",
    ".resizespike.dvt",
    ".compress.eventframe.bin",
    ".compress.event.bin",
    ".txt"};

void new_dpu_pic(DPU_COMMON *common, DPU_PIC_TYPE pic_type, int w, int h);
void memcpy_dpu_pic(DPU_PIC *src, DPU_PIC *dst);
void memset_dpu_pic(DPU_PIC *pic, int value);
void memset_v2e_pic(DPU_PIC_2 *pic, int value);
void write_frame_raw(DPU_PIC *pic, FILE *fp);
void write_frame_ppm(DPU_PIC *pic, FILE *fp, int i, char id);
void read_frame_raw(DPU_PIC *pic, FILE *fp);
int read_frame_rgb(DPU_PIC *pic, FILE *fp);

void allocate_buffer_accdenoise(DPU_INTERFACE *interface, DPU_COMMON *common);
void allocate_buffer_v2e(DPU_INTERFACE *interface, DPU_COMMON *common);
void allocate_buffer_resize(DPU_INTERFACE *interface, DPU_COMMON *common);
void allocate_buffer_event(DPU_COMMON *common);
void allocate_buffer_ef2spike(DPU_COMMON *common);
void allocate_buffer_eventcount(DPU_COMMON *common);
void allocate_buffer_roi(DPU_COMMON *common);
void allocate_buffer_eventframecrop(DPU_COMMON *common);
void allocate_buffer_EVT3(DPU_COMMON *common);

void read_interface(DPU_INTERFACE *it, DPU_COMMON *cm);
void write_interface(DPU_INTERFACE *it, DPU_COMMON *cm);

void read_DAT_header(DPU_COMMON *common);
int read_DAT_event(DPU_COMMON *common);

void read_EVT3_header(DPU_COMMON *common);
int read_EVT3_event(DPU_COMMON *common);

void write_DPU_event(DPU_COMMON *common);
int read_DPU_event(DPU_COMMON *common);
void new_v2e_pic(DPU_COMMON *common, DPU_PIC_TYPE pic_type, int w, int h);

const char *get_savepath(DPU_COMMON *cm, const char *folder, MODULES module);

void memreset_eventcount(DPU_INTERFACE *interface, DPU_COMMON *common);
void memreset_eventframecrop(DPU_INTERFACE *interface, DPU_COMMON *common);
void memreset_event(DPU_INTERFACE *interface, DPU_COMMON *common);
void memreset_roi(DPU_INTERFACE *interface, DPU_COMMON *common);
void memreset_evt(DPU_COMMON *common);
void memreset_ef2spike(DPU_INTERFACE *interface, DPU_COMMON *common);

void free_all(DPU_INTERFACE *interface, DPU_COMMON *common);

void ACC_MAP(DPU_INTERFACE *interface, DPU_COMMON *cm);
void DAT2DVT(DPU_INTERFACE *interface, DPU_COMMON *common);
void DAT2DVT_EVT3(DPU_INTERFACE *interface, DPU_COMMON *common, int idx);

int32_t average_int(DPU_INTERFACE *interface, uint8_t *src, uint32_t src_stride);
int32_t square_average_int(DPU_INTERFACE *interface, uint8_t *src, uint32_t src_stride);
// void DENOISE(DPU_INTERFACE *interface, DPU_COMMON *cm);

void ACC_HIS(DPU_INTERFACE *interface, DPU_COMMON *cm);

void SIZE_EVENTFRAME(DPU_INTERFACE *interface, DPU_COMMON *cm);
void CROP_EVENTFRAME(DPU_INTERFACE *interface, DPU_COMMON *cm);
void WRITE_CROP_EVENTFRAME(DPU_INTERFACE *interface, DPU_COMMON *cm);
#endif
