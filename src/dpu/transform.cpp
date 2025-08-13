// 아르고 코드 그냥 다 넣음
#include "transform.h"
#include "variables.h"
#include "utils.h"

using argo_vars::eps_list;
using argo_vars::lookuptable;
using argo_vars::mean_scale;

inline uint8_t CLIP3_km(int val)
{
    if (val < 0)
        return 0;
    if (val > 255)
        return 255;
    return (uint8_t)val;
}

void ACC_MAP(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    if (!interface->acc_map_on)
        return;

    DPU_EVENT *dpu = &cm->dpu_event;
    DPU_PIC *pic = &cm->pic[PIC_TYPE_ACC];

    int x = dpu->X;
    int y = dpu->Y;
    int p = dpu->polarity ? +1 : -1;

    int32_t offset = y * pic->stride + x;
    uint8_t *buf = pic->buf + offset;

    // *buf = CLIP3((int)(*buf) + (int)(p < 0 ? -1 : +1) * (abs(p) << interface->linear_map_log2scale)); // 4만큼 shift
    int delta = (p ? 1 : -1) << interface->linear_map_log2scale;
    *buf = CLIP3_km((int)(*buf) + delta);
}

void memcpy_dpu_pic(DPU_PIC *src, DPU_PIC *dst)
{
    uint8_t *src_buf = src->buf;
    uint8_t *dst_buf = dst->buf;
    for (int i = 0; i < src->height; i++)
    {
        memcpy(dst_buf, src_buf, sizeof(uint8_t) * src->width);
        src_buf += src->stride;
        dst_buf += dst->stride;
    }
}

void memset_dpu_pic(DPU_PIC *pic, int value)
{
    memset(pic->buf_origin, value, sizeof(uint8_t) * BUFFER_SIZE);
}

void memset_v2e_pic(DPU_PIC_2 *pic, int value)
{
    memset(pic->buf_origin, value, sizeof(uint16_t) * BUFFER_SIZE);
}

void new_dpu_pic(DPU_COMMON *common, DPU_PIC_TYPE pic_type, int w, int h)
{
    DPU_PIC *pic = &common->pic[pic_type];
    pic->buf_origin = common->pic_pool + pic_type * BUFFER_SIZE;
    pic->buf = pic->buf_origin + BUFFER_WIDTH * (PADDING >> 1) + (PADDING >> 1);
    pic->stride = BUFFER_WIDTH;
    pic->width = w;
    pic->height = h;
    memset_dpu_pic(pic, 0); // 1101
}
void new_v2e_pic(DPU_COMMON *common, DPU_PIC_TYPE pic_type, int w, int h)
{
    DPU_PIC_2 *pic2 = &common->pic2[pic_type];
    pic2->buf_origin = common->pic_v2e_pool + pic_type * BUFFER_SIZE;
    pic2->buf = pic2->buf_origin + BUFFER_WIDTH * (PADDING >> 1) + (PADDING >> 1);
    pic2->stride = BUFFER_WIDTH;
    pic2->width = w;
    pic2->height = h;
    memset_v2e_pic(pic2, 0); // 1101
}

void allocate_buffer_accdenoise(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    common->pic_pool = (uint8_t *)malloc(sizeof(uint8_t) * BUFFER_SIZE * BUFFER_NUM);
    new_dpu_pic(common, PIC_TYPE_ACC, interface->input_width, interface->input_height);
    new_dpu_pic(common, PIC_TYPE_ACC_PREV, interface->input_width, interface->input_height);
    new_dpu_pic(common, PIC_TYPE_EVT, interface->input_width, interface->input_height);
}

void allocate_buffer_v2e(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    common->pic_pool = (uint8_t *)malloc(sizeof(uint8_t) * BUFFER_SIZE * BUFFER_NUM);
    common->pic_v2e_pool = (uint16_t *)malloc(sizeof(uint16_t) * BUFFER_SIZE * BUFFER_NUM);
    new_dpu_pic(common, PIC_TYPE_EVT, interface->input_width, interface->input_height);
    new_v2e_pic(common, PIC_TYPE_RGB_PREV, interface->input_width, interface->input_height);
    new_dpu_pic(common, PIC_TYPE_R, interface->input_width, interface->input_height);
    new_dpu_pic(common, PIC_TYPE_G, interface->input_width, interface->input_height);
    new_dpu_pic(common, PIC_TYPE_B, interface->input_width, interface->input_height);
}

void allocate_buffer_resize(DPU_INTERFACE *interface, DPU_COMMON *common)
{

    common->pic_resize_pool = (uint8_t *)malloc(sizeof(uint8_t) * (64 * 64));
    memset(common->pic_resize_pool, 0, sizeof(uint8_t) * (64 * 64));
}

void allocate_buffer_event(DPU_COMMON *common)
{
    common->dpu_crop.dpu_store_buf = (DPU_EVENT *)malloc(sizeof(DPU_EVENT) * (DPU_STORE_SIZE));
    common->dpu_crop.dpu_crop_buf = (DPU_EVENT *)malloc(sizeof(DPU_EVENT) * (DPU_STORE_SIZE));
    common->dpu_crop.offset_buf = (uint16_t *)malloc(sizeof(uint16_t) * 15);
    common->dpu_crop.count_buf = (uint16_t *)malloc(sizeof(uint16_t) * 15);
    memset(common->dpu_crop.dpu_store_buf, 0, sizeof(DPU_EVENT) * DPU_STORE_SIZE);
    memset(common->dpu_crop.dpu_crop_buf, 0, sizeof(DPU_EVENT) * DPU_STORE_SIZE);
    memset(common->dpu_crop.offset_buf, 0, sizeof(uint16_t) * 15);
    memset(common->dpu_crop.count_buf, 0, sizeof(uint16_t) * 15);
}

void allocate_buffer_ef2spike(DPU_COMMON *common)
{
    common->frame_event = (DAT_EVENT *)malloc(sizeof(DAT_EVENT) * (FRAME_SIZE) * (303));
    common->frame_event_count = (uint16_t *)malloc(sizeof(uint16_t) * (FRAME_SIZE) * (303));
    memset(common->frame_event, 2, sizeof(DAT_EVENT) * (FRAME_SIZE) * (303));
    memset(common->frame_event_count, 0, sizeof(uint16_t) * (FRAME_SIZE) * (303));
}

void allocate_buffer_eventcount(DPU_COMMON *common)
{
    common->histogram.histogram_y = (uint16_t *)malloc(sizeof(uint16_t) * (704 >> 6));
    common->histogram.histogram_x = (uint16_t *)malloc(sizeof(uint16_t) * (1280 >> 6));
    memset(common->histogram.histogram_y, 0, sizeof(uint16_t) * (704 >> 6));
    memset(common->histogram.histogram_x, 0, sizeof(uint16_t) * (1280 >> 6));
}
void allocate_buffer_roi(DPU_COMMON *common)
{
    common->roi.roi_buf = (uint16_t *)malloc(sizeof(uint16_t) * 60);
    memset(common->roi.roi_buf, 0, sizeof(uint16_t) * (60));
    common->roi.static_roix = (uint16_t *)malloc(sizeof(uint16_t) * 3);
    common->roi.static_roiy = 192;
    common->roi.static_roix[0] = 0;
    common->roi.static_roix[1] = 448;
    common->roi.static_roix[2] = 896;
}

void allocate_buffer_eventframecrop(DPU_COMMON *common)
{
    common->event_frame_crop_buf_size = (uint32_t *)malloc(sizeof(uint32_t) * 15);
    common->event_frame_crop_buf = (uint8_t *)malloc(sizeof(uint8_t) * BUFFER_SIZE);
    memset(common->event_frame_crop_buf_size, 0, sizeof(uint32_t) * 15);
    memset(common->event_frame_crop_buf, 0, sizeof(uint8_t) * BUFFER_SIZE);
}

void allocate_buffer_EVT3(DPU_COMMON *common)
{
    common->dat_event_12 = (DAT_EVENT *)malloc(sizeof(DAT_EVENT) * 12);
}

void memreset_eventframecrop(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    if (interface->eventframecrop_on)
    {
        memset(common->event_frame_crop_buf_size, 0, sizeof(uint32_t) * 15);
        memset(common->event_frame_crop_buf, 0, sizeof(uint8_t) * BUFFER_SIZE);
    }
}

void memreset_eventcount(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    if (interface->eventcount_on)
    {
        memset(common->histogram.histogram_y, 0, sizeof(uint16_t) * (704 >> 6));
        memset(common->histogram.histogram_x, 0, sizeof(uint16_t) * (1280 >> 6));
    }
}
void memreset_event(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    if (interface->eventcrop_on)
    {
        memset(common->dpu_crop.dpu_store_buf, 0, sizeof(DPU_EVENT) * DPU_STORE_SIZE);
        memset(common->dpu_crop.dpu_crop_buf, 0, sizeof(DPU_EVENT) * DPU_STORE_SIZE);
        memset(common->dpu_crop.offset_buf, 0, sizeof(uint16_t) * 15);
        memset(common->dpu_crop.count_buf, 0, sizeof(uint16_t) * 15);
    }
}
void memreset_roi(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    if (interface->roi_on)
        memset(common->roi.roi_buf, 0, sizeof(uint16_t) * (60));
}
void memreset_ef2spike(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    if (interface->ef2spike_on)
        memset(common->frame_event, 2, sizeof(DAT_EVENT) * (FRAME_SIZE) * (303));
}
void memreset_evt(DPU_COMMON *common)
{
    memset(common->dat_event_12, 0, sizeof(uint64_t) * 12);
}

void free_all(DPU_INTERFACE *interface, DPU_COMMON *common)
{
    // event 관련 buf 해제

    if (interface->resize_on)
    {
        free(common->pic_resize_pool); // rgb
    }
    // free(common->dpu_crop.dpu_store_buf); //dat
    free(common->dpu_crop.dpu_crop_buf);
    // free(common->dpu_crop.offset_buf);//rgb dat
    free(common->dpu_crop.count_buf);

    if (interface->ef2spike_on)
    {
        free(common->frame_event);
        free(common->frame_event_count);
    }
    // roi 및 histogram buf 해제
    free(common->roi.roi_buf);
    free(common->roi.static_roix);
    if (interface->eventcount_on)
    {
        free(common->histogram.histogram_y);
        free(common->histogram.histogram_x);
    }
    // eventframe crop 관련 buf 해제
    if (interface->eventframecrop_on)
    {
        free(common->event_frame_crop_buf);
        free(common->event_frame_crop_buf_size);
    }
    // dat_event_12 buf 해제
    if (interface->evt_format)
        free(common->dat_event_12);
    free(common->pic_pool);
    if (interface->video2eventframe_on)
        free(common->pic_v2e_pool);
}

void write_frame_raw(DPU_PIC *pic, FILE *fp)
{
    uint8_t *buf = pic->buf;

    for (int i = 0; i < pic->height; i++)
    {
        fwrite(buf, sizeof(uint8_t) * pic->width, 1, fp);
        buf += pic->stride;
    }
}

void write_frame_ppm(DPU_PIC *pic, FILE *fp, int i, char id)
{
    uint8_t *buf = pic->buf;
    char ppmtxt[1000];
    char ppmf[1000];

    strcpy(ppmtxt, "P5\n1280\n720\n255\n");

    if (id == 0)
    {
        sprintf(ppmf, "./evt_sample/ACC_MAP/acc_map_%d.ppm", i);
    }
    else
    {
        sprintf(ppmf, "./evt_sample/test_%d.ppm", i);
    }

    fp = fopen(ppmf, "wb");
    fwrite(ppmtxt, sizeof(char), 16, fp);

    for (int i = 0; i < pic->height; i++)
    {
        fwrite(buf, sizeof(uint8_t) * pic->width, 1, fp);
        buf += pic->stride;
    }
    fclose(fp);
}

int read_frame_rgb(DPU_PIC *pic, FILE *fp)
{
    if (0 != feof(fp))
        return 0;
    uint8_t *buf = pic->buf;
    for (int i = 0; i < pic->height; i++)
    {
        fread(buf, sizeof(uint8_t) * pic->width, 1, fp);
        buf += pic->stride;
    }
    return 1;
}

void read_interface(DPU_INTERFACE *it, DPU_COMMON *cm)
{
    char str[1000];
    FILE *fp = fopen(strcat(strcpy(str, cm->wkdir), "/interface.bin"), "rb");
    fread(it, sizeof(DPU_INTERFACE), 1, fp);
    fclose(fp);
}
void write_interface(DPU_INTERFACE *it, DPU_COMMON *cm)
{
    char str[1000];
    FILE *fp = fopen(strcat(strcpy(str, cm->wkdir), "/interface.bin"), "wb");
    fwrite(it, sizeof(DPU_INTERFACE), 1, fp);
    fclose(fp);
}

void read_DAT_header(DPU_COMMON *common)
{
    int endOfHeader = 0;
    unsigned int bod = 0;
    char tline[10000];

    while (!endOfHeader)
    {
        bod = ftell(common->fp_dat);
        fgets(tline, 256, common->fp_dat);
        if (tline[0] != '%')
            endOfHeader = 1;
    }

    fseek(common->fp_dat, bod, SEEK_SET);

    // event type, size parsing
    char eventType = 0;
    char eventSize = 0;

    fread(&eventType, sizeof(char), 1, common->fp_dat);
    fread(&eventSize, sizeof(char), 1, common->fp_dat);

    long bof = ftell(common->fp_dat);
    fseek(common->fp_dat, bof, SEEK_SET);
}

void read_EVT3_header(DPU_COMMON *common)
{
    int endOfHeader = 0;
    unsigned int bod = 0;
    char tline[10000];

    while (!endOfHeader)
    {
        bod = ftell(common->fp_dat);
        fgets(tline, 256, common->fp_dat);
        if (tline[0] != '%')
        {
            endOfHeader = 1;
            break;
        }
    }
    fseek(common->fp_dat, bod, SEEK_SET);
    long bof = ftell(common->fp_dat);
    fseek(common->fp_dat, bof, SEEK_SET);
}

int read_EVT3_event(DPU_COMMON *common)
{

    int ret = fread(&(common->raw_event), 1, 2, common->fp_dat);

    if (ret == 0)
        return 0;

    EventTypes type = (EventTypes)(common->raw_event.type);

    if (!(common->first_time_base_set) && type == EVT_TIME_HIGH)
    {
        RawEventTime *ev_timehigh = (RawEventTime *)(&(common->raw_event));
        common->current_time_base = ((timestamp_t)(ev_timehigh->time) << 12);
        common->first_time_base_set = true;

        return 1;
    }

    if (!(common->first_time_base_set) && type != EVT_TIME_HIGH)
    {
        return 1;
    }

    if (type == EVT_TIME_HIGH)
    {
        timestamp_t MaxTimestampBase =
            (((timestamp_t)(1) << 12) - 1) << 12;            // = 16773120us
        timestamp_t TimeLoop = MaxTimestampBase + (1 << 12); // = 16777216us
        timestamp_t LoopThreshold = (10 << 12);              // It could be another value too, as long as it is a big enough value that we can be
                                                             // sure that the time high looped

        RawEventTime *ev_timehigh = (RawEventTime *)(&(common->raw_event));

        timestamp_t new_time_base = ((timestamp_t)(ev_timehigh->time) << 12);
        new_time_base += (common->n_time_high_loop) * TimeLoop;

        if ((common->current_time_base > new_time_base) &&
            (common->current_time_base - new_time_base >= MaxTimestampBase - LoopThreshold))
        {
            // Time High loop :  we consider that we went in the past because the timestamp looped
            new_time_base += TimeLoop;
            ++(common->n_time_high_loop);
        }

        common->current_time_base = new_time_base;
        common->current_time = common->current_time_base;

        return 1;
    }

    else if (type == X_POS)
    {
        RawEventXPos *ev_cd_posx = (RawEventXPos *)(&(common->raw_event));
        common->current_x_base = ev_cd_posx->x; // X_POS also updates the X_BASE

        if (common->current_type == CD)
        {
            (common->dat_event_12)[0].X = common->current_x_base;
            (common->dat_event_12)[0].Y = common->current_cd_y;
            (common->dat_event_12)[0].timestamp = common->current_time;
            (common->dat_event_12)[0].polarity = ev_cd_posx->pol;
        }
        return 2;
    }

    else if (type == VECT_12)
    {
        uint16_t end = common->current_x_base + 12;
        int idx_evt = 0;

        if (common->current_type == CD)
        {
            RawEventVect12 *ev_vec_12 = (RawEventVect12 *)(&(common->raw_event));
            uint32_t valid = ev_vec_12->valid;

            for (uint16_t i = common->current_x_base; i != end; ++i)
            {
                if (valid & 0x1)
                {
                    (common->dat_event_12)[idx_evt].X = i;
                    (common->dat_event_12)[idx_evt].Y = common->current_cd_y;
                    (common->dat_event_12)[idx_evt].timestamp = common->current_time;
                    (common->dat_event_12)[idx_evt].polarity = common->current_polarity;
                    idx_evt += 1;
                }
                valid >>= 1;
            }
        }
        common->current_x_base = end;

        return idx_evt + 1;
    }

    else if (type == VECT_8)
    {
        uint16_t end = common->current_x_base + 8;
        int idx_evt = 0;

        if (common->current_type == CD)
        {
            RawEventVect8 *ev_vec_8 = (RawEventVect8 *)(&(common->raw_event));
            uint32_t valid = ev_vec_8->valid;

            for (uint16_t i = common->current_x_base; i != end; ++i)
            {
                if (valid & 0x1)
                {
                    (common->dat_event_12)[idx_evt].X = i;
                    (common->dat_event_12)[idx_evt].Y = common->current_cd_y;
                    (common->dat_event_12)[idx_evt].timestamp = common->current_time;
                    (common->dat_event_12)[idx_evt].polarity = common->current_polarity;
                    idx_evt += 1;
                }
                valid >>= 1;
            }
        }
        common->current_x_base = end;

        return idx_evt + 1;
    }

    else if (type == CD_Y)
    {
        common->current_type = CD;

        RawEventY *ev_cd_y = (RawEventY *)(&(common->raw_event));
        common->current_cd_y = ev_cd_y->y;

        return 1;
    }

    else if (type == EM_Y)
    {
        common->current_type = EM;

        return 1;
    }

    else if (type == X_BASE)
    {
        RawEventXBase *ev_xbase = (RawEventXBase *)(&(common->raw_event));
        common->current_polarity = ev_xbase->pol;
        common->current_x_base = ev_xbase->x;

        return 1;
    }

    else if (type == EVT_TIME_LOW)
    {
        RawEventTime *ev_timelow = (RawEventTime *)(&(common->raw_event));
        common->current_time_low = ev_timelow->time;
        common->current_time = common->current_time_base + common->current_time_low;

        return 1;
    }
}

int read_DAT_event(DPU_COMMON *common)
{
    return fread(&common->dat_event, 1, 8, common->fp_dat);
}

void write_DPU_event(DPU_COMMON *common)
{
    fwrite(&common->dpu_event, sizeof(DPU_EVENT), 1, common->fp_dpu);
    // fwrite(&common->dat_event, sizeof(DAT_EVENT), 1, common->fp_dpu );
}

int read_DPU_event(DPU_COMMON *common)
{
    return fread(&common->dpu_event, sizeof(DPU_EVENT), 1, common->fp_dpu);
}

const char *get_savepath(DPU_COMMON *cm, const char *folder, MODULES module)
{
    static char str[100] = "\0";
    strcat(strcat(strcat(strcpy(str, cm->dirs[module]), "/"), folder), module_ext[module]);
    return str;
}

void DAT2DVT(DPU_INTERFACE *interface, DPU_COMMON *common) // DAT 변환
{
    if (!interface->evt_cvt_on)
        return;
    common->dpu_event.timestamp = common->dat_event.timestamp / 1111; //  1.1 ms
    common->dpu_event.polarity = common->dat_event.polarity & 0x0001;
    common->dpu_event.X = common->dat_event.X & 0x0FFF;
    common->dpu_event.Y = common->dat_event.Y & 0x07FF;
}

void DAT2DVT_EVT3(DPU_INTERFACE *interface, DPU_COMMON *common, int idx) // EVT 변환
{
    if (!interface->evt_cvt_on)
        return;
    common->dpu_event.timestamp = (common->dat_event_12)[idx].timestamp / 1111; //  1.1ms
    common->dpu_event.polarity = (common->dat_event_12)[idx].polarity & 0x0001;
    common->dpu_event.X = (common->dat_event_12)[idx].X & 0x0FFF;
    common->dpu_event.Y = (common->dat_event_12)[idx].Y & 0x07FF;
}

void ACC_HIS(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    // x,y histogram 생성
    if (interface->eventcount_on && cm->dpu_event.Y >= 8 && cm->dpu_event.Y < 712)
    {
        cm->histogram.histogram_y[(cm->dpu_event.Y - 8) >> 6] += 1;
        cm->histogram.histogram_x[cm->dpu_event.X >> 6] += 1;
    }
}

void SIZE_EVENTFRAME(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    // cropped event frmae size calculates
    if (interface->eventframecrop_on)
    {

        uint32_t *evf_crop_size = cm->event_frame_crop_buf_size;
        uint16_t *roi_buf = cm->roi.roi_buf;
        int i;
        for (i = 0; i < cm->roi.roi_count; i++)
        {
            *(evf_crop_size + i) = (*(roi_buf + (i << 2) + 2) - *(roi_buf + (i << 2)) + 1);
        }
    }
}

void CROP_EVENTFRAME(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    // Event Frame crop
    if (interface->eventframecrop_on)
    {
        DPU_PIC *pic = &cm->pic[PIC_TYPE_EVT];
        uint8_t *frame_crop_buf = cm->event_frame_crop_buf;
        uint32_t *frame_crop_buf_size = cm->event_frame_crop_buf_size;
        uint16_t *roi_buf = cm->roi.roi_buf;
        uint8_t *buf_ptr;
        int32_t offset;
        uint16_t init_x, init_y, end_x, end_y;
        int idx = 0;
        int idx_curr_ROI;

        for (idx_curr_ROI = 0; idx_curr_ROI < (cm->roi.roi_count); idx_curr_ROI++)
        {
            init_x = *(cm->roi.roi_buf + (idx_curr_ROI << 2));
            init_y = *(cm->roi.roi_buf + (idx_curr_ROI << 2) + 1);
            end_x = *(cm->roi.roi_buf + (idx_curr_ROI << 2) + 2);
            end_y = *(cm->roi.roi_buf + (idx_curr_ROI << 2) + 3);

            offset = pic->stride * init_y + init_x;
            buf_ptr = pic->buf + offset;

            for (int i = init_y; i <= end_y; i++)
            {
                for (int j = init_x; j <= end_x; j++)
                {
                    *(frame_crop_buf + idx) = *(buf_ptr);
                    buf_ptr += 1;
                    idx++;
                }
                buf_ptr = buf_ptr - (end_x - init_x + 1) + pic->stride;
            }
        }
    }
}
void WRITE_CROP_EVENTFRAME(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    // Cropped Event Frame write
    if (interface->eventframecrop_on)
    {
        uint16_t *roi_buf = cm->roi.roi_buf;
        uint8_t *frame_crop_buf = cm->event_frame_crop_buf;
        uint32_t *frame_crop_buf_size = cm->event_frame_crop_buf_size;

        int idx_curr_ROI;
        char frame_count_str[1000];
        char idx_curr_ROI_str[1000];
        char ROI_width_str[1000];
        char ROI_height_str[1000];
        char crop_eventframe[1000];
        sprintf(frame_count_str, "%d", cm->frame_count);

        for (idx_curr_ROI = 0; idx_curr_ROI < (cm->roi.roi_count); idx_curr_ROI++)
        {
            sprintf(idx_curr_ROI_str, "%d", idx_curr_ROI);
            sprintf(ROI_width_str, "%d", (*(roi_buf + (idx_curr_ROI << 2) + 2)) - (*(roi_buf + (idx_curr_ROI << 2))) + 1);
            sprintf(ROI_height_str, "%d", (*(roi_buf + (idx_curr_ROI << 2) + 3)) - (*(roi_buf + (idx_curr_ROI << 2) + 1)) + 1);
            strcpy(crop_eventframe, cm->filename_roi_eventframe);
            strcat(crop_eventframe, "_");
            strcat(crop_eventframe, frame_count_str);
            strcat(crop_eventframe, "_");
            strcat(crop_eventframe, idx_curr_ROI_str);
            strcat(crop_eventframe, "_");
            strcat(crop_eventframe, ROI_width_str);
            strcat(crop_eventframe, "_");
            strcat(crop_eventframe, ROI_height_str);
            strcat(crop_eventframe, ".raw");
            cm->fp_crop_eventframe = fopen(crop_eventframe, "wb");
            if (idx_curr_ROI > 0)
            {
                fwrite(frame_crop_buf, sizeof(uint8_t), (*(frame_crop_buf_size + idx_curr_ROI)) * (*(frame_crop_buf_size + idx_curr_ROI)), cm->fp_crop_eventframe);
                frame_crop_buf += (*(frame_crop_buf_size + idx_curr_ROI)) * (*(frame_crop_buf_size + idx_curr_ROI));
            }
            else if (idx_curr_ROI == 0)
            {
                fwrite(frame_crop_buf, sizeof(uint8_t), (*(frame_crop_buf_size)) * (*(frame_crop_buf_size)), cm->fp_crop_eventframe);
                frame_crop_buf += (*(frame_crop_buf_size)) * (*(frame_crop_buf_size));
            }
            fclose(cm->fp_crop_eventframe);
        }
    }
}

int32_t average_int(DPU_INTERFACE *interface, uint8_t *src, uint32_t src_stride)
{
    const int r = interface->filter_radius;
    const int scale = argo_vars::mean_scale[r - 1];
    const int tap = 2 * r + 1;
    const int base_offset = -r * (int)src_stride - r;

    int32_t sum = 0;
    for (int p = 0; p < tap; ++p)
    {
        int row_offset = base_offset + p * src_stride;
        for (int q = 0; q < tap; ++q)
        {
            sum += scale * src[row_offset + q];
        }
    }

    return (sum + (1 << 14)) >> 15;
}

int32_t square_average_int(DPU_INTERFACE *interface, uint8_t *src, uint32_t src_stride)
{
    const int r = interface->filter_radius;
    const int scale = argo_vars::mean_scale[r - 1];
    const int tap = 2 * r + 1;
    const int base_offset = -r * (int)src_stride - r;

    int32_t sum = 0;
    for (int p = 0; p < tap; ++p)
    {
        int row_offset = base_offset + p * src_stride;
        for (int q = 0; q < tap; ++q)
        {
            int v = src[row_offset + q];
            sum += scale * v * v;
        }
    }

    return (sum + (1 << 14)) >> 15;
}