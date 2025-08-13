#include "core.h"
#include "utils.h"

void init_interface(DPU_INTERFACE *interface, int width, int height, int linear_map_log2scale, int eps_idx, int filter_radius)
{

    // COMMONs
    interface->input_width = width;   // todo: 이거 입력받기
    interface->input_height = height; // todo: 이거 입력받기
    interface->bitdepth = FRAME_BITDEPTH;
    // INPUT EVENT FORMAT
    interface->evt_format = 1; // 0 : DAT, 1 : EVT3.0
    // DAT2DVT
    interface->evt_cvt_on = 1; // dat2dvt 모듈 ON/OFF 0 : OFF, 1 : ON
    // ACC_MAP & DENOISE
    interface->acc_map_on = 1;                              // acc_map & denoise 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->linear_map_log2scale = linear_map_log2scale; // scaling //todo: 이거 입력받기
    interface->eps_idx = eps_idx;                           // denoise 모듈의 regulatization parameter 0 : 0.05, 1: 0.1, 2 : 0.15, 3 : 0.2 //todo: 이거 입력받기
    interface->filter_radius = filter_radius;               // denoise 모듈의 필터 크기 0 : 1, 1 : 2, 2 : 3, 3 : 4 //todo: 이거 입력받기
    // COMPRESS
    interface->eventframe_compress_on = 0; // eventframe_compress 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->event_compress_on = 0;      // event_compress 모듈 ON/OFF 0 : OFF, 1 : ON
    // EVENTCOUNT
    interface->eventcount_on = 0; // eventcount 모듈 ON/OFF 0 : OFF, 1 : ON
    // CROP
    interface->eventframecrop_on = 0; // eventframecrop 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->eventcrop_on = 0;      // eventcrop 모듈 ON/OFF 0 : OFF, 1 : ON
    // ROI
    interface->roi_on = 0;          // roi 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->static_roi_flag = 0; // static_roi 사용하는지 안하는지 0 : 사용안함, 1 : 사용함
    // AUTO EXPOSURE
    interface->autoexposure_on = 0; // autoexposure 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->max_acc_time = 2;    // 최대 반복 누적 횟수 2번이 최대 3.3ms까지 0 : 0번, 1: 1번, 2: 2번
    // V2E
    interface->video2eventframe_on = 0; // video2eventframe 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->v2e_threshold = 1;       // v2e_threshold 밝기 값 변화 트리거에 대한 임계값 = 0 : 0.1, 1 : 0.15, 2 : 0.2, 3 : 0.25, 4 : 0.3
    // RESIZE
    interface->resize_on = 0;           // resize 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->downsampling_method = 0; // 이벤트 다운샘플링 방법 0 : pixel subsampling, 1 : bilinear, 2 : bicubic
    interface->input_type = 0;          // resize 모듈의 입력 0: event frame, 1 : event, 2: frame에서 변환된 event
    // EVENT FRAME TO SPIKE
    interface->ef2spike_on = 0;  // ef2spike 모듈 ON/OFF 0 : OFF, 1 : ON
    interface->spike_method = 1; // 스파이크 변환 방법 0: 균등분포, 1 : 푸아송분포
}

const char *get_dpu_savepath(DPU_COMMON *cm, const char *folder, MODULES module)
{
    static char str[100] = "\0";
    strcat(strcat(strcat(strcat(strcpy(str, cm->dirs[module]), "/"), folder), "_"), module_ext[module]);
    return str;
}

void init_dpu(DPU_INTERFACE *interface, DPU_COMMON *common, int width, int height, int linear_map_log2scale, int eps_idx, int filter_radius)
{
    //! [kkm] 기존 코드에서 파일 관련 코드 제거

    // char str[100] = "\0";
    // char folder[100] = "\0";
    // char *temp;

    // int st = 0, ed = 0;
    // for (int i = strlen(filename) - 1; i >= 0; i--)
    // {
    //     if (filename[i] == '.')
    //         ed = i;
    //     if (filename[i] == '/')
    //     {
    //         st = i;
    //         break;
    //     }
    // }
    // for (int i = st + 1; i < ed; i++)
    // {
    //     folder[i - (st + 1)] = filename[i];
    // }
    // folder[ed - (st + 1)] = '\0';

    // temp = strcat(strcat(str, "./"), folder);
    // strcpy(common->wkdir, temp);
    // mkdir(common->wkdir, 0777);
    // strcpy(common->dirs[MODULE_ACC_MAP], temp);
    // mkdir(strcat(common->dirs[MODULE_ACC_MAP], "/ACC_MAP"), 0777);
    // strcpy(common->dirs[MODULE_HDEVENTFRAME], temp);
    // mkdir(strcat(common->dirs[MODULE_HDEVENTFRAME], "/HDEVENTFRAME"), 0777);
    // strcpy(common->dirs[MODULE_ROIEVENTFRAME], temp);
    // mkdir(strcat(common->dirs[MODULE_ROIEVENTFRAME], "/ROIEVENTFRAME"), 0777);
    // strcpy(common->dirs[MODULE_RESIZEEVENTFRAME], temp);
    // mkdir(strcat(common->dirs[MODULE_RESIZEEVENTFRAME], "/RESIZEEVENTFRAME"), 0777);
    // strcpy(common->dirs[MODULE_FRAME_HDSPIKE], temp);
    // mkdir(strcat(common->dirs[MODULE_FRAME_HDSPIKE], "/FRAME_HDEVENT"), 0777);
    // strcpy(common->dirs[MODULE_FRAME_ROISPIKE], temp);
    // mkdir(strcat(common->dirs[MODULE_FRAME_ROISPIKE], "/FRAME_ROIEVENT"), 0777);
    // strcpy(common->dirs[MODULE_FRAME_RESIZESPIKE], temp);
    // mkdir(strcat(common->dirs[MODULE_FRAME_RESIZESPIKE], "/FRAME_RESIZEEVENT"), 0777);
    // strcpy(common->dirs[MODULE_HDSPIKE], temp);
    // mkdir(strcat(common->dirs[MODULE_HDSPIKE], "/HDEVENT"), 0777);
    // strcpy(common->dirs[MODULE_ROISPIKE], temp);
    // mkdir(strcat(common->dirs[MODULE_ROISPIKE], "/ROIEVENT"), 0777);
    // strcpy(common->dirs[MODULE_RESIZESPIKE], temp);
    // mkdir(strcat(common->dirs[MODULE_RESIZESPIKE], "/RESIZEEVENT"), 0777);
    // strcpy(common->dirs[MODULE_COMPRESS_EVENTFRAME], temp);
    // mkdir(strcat(common->dirs[MODULE_COMPRESS_EVENTFRAME], "/COMPRESS_EVENTFRAME"), 0777);
    // strcpy(common->dirs[MODULE_COMPRESS_EVENT], temp);
    // mkdir(strcat(common->dirs[MODULE_COMPRESS_EVENT], "/COMPRESS_EVENT"), 0777);
    // strcpy(common->dirs[MODULE_TXT], temp);
    // mkdir(strcat(common->dirs[MODULE_TXT], "/TXT"), 0777);
    // init interface
    init_interface(interface, width, height, linear_map_log2scale, eps_idx, filter_radius);
    // write_interface(interface, common);

    // set file pointers
    // if (interface->video2eventframe_on)
    //     common->fp_rgb = fopen(argv[1], "rb");
    // else
    //     common->fp_dat = fopen(argv[1], "rb");
    // common->fp_dpu = fopen(get_dpu_savepath(common, folder, MODULE_HDSPIKE), "wb");
    // common->fp_accmap = fopen(get_dpu_savepath(common, folder, MODULE_ACC_MAP), "wb");
    // common->fp_hdeventframe = fopen(get_dpu_savepath(common, folder, MODULE_HDEVENTFRAME), "wb");
    // if (interface->ef2spike_on)
    //     common->fp_framehd_dpu = fopen(get_dpu_savepath(common, folder, MODULE_FRAME_HDSPIKE), "wb");

    // strcat(strcat(strcpy(common->filename_bitstream_eventframe, common->dirs[MODULE_COMPRESS_EVENTFRAME]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_bitstream_event, common->dirs[MODULE_COMPRESS_EVENT]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_roi_event, common->dirs[MODULE_ROISPIKE]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_roi_eventframe, common->dirs[MODULE_ROIEVENTFRAME]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_resize_event, common->dirs[MODULE_RESIZESPIKE]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_resize_eventframe, common->dirs[MODULE_RESIZEEVENTFRAME]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_frame_resize_event, common->dirs[MODULE_FRAME_RESIZESPIKE]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_frame_roi_event, common->dirs[MODULE_FRAME_ROISPIKE]), "/"), folder);
    // strcat(strcat(strcpy(common->filename_roi_txt, common->dirs[MODULE_TXT]), "/"), folder);

    // // buffer allocation
    if (interface->acc_map_on)
        allocate_buffer_accdenoise(interface, common);
    if (interface->evt_format)
        allocate_buffer_EVT3(common);
    if (interface->video2eventframe_on)
        allocate_buffer_v2e(interface, common);
    if (interface->resize_on)
        allocate_buffer_resize(interface, common);
    if (interface->ef2spike_on)
        allocate_buffer_ef2spike(common);
    if (interface->eventcount_on)
        allocate_buffer_eventcount(common);
    if (interface->eventframecrop_on)
        allocate_buffer_eventframecrop(common);
    allocate_buffer_event(common);
    allocate_buffer_roi(common);

    // init variables
    common->frame_count = 0;
    common->prev_ts = 0;
    common->evt_count = 0;
    common->prv_evt_count = 0;
    common->windowcount = 0;
    common->current_time_base = 0;
    common->current_time_low = 0;
    common->current_time = 0;
    common->current_cd_y = 0;
    common->current_x_base = 0;
    common->current_polarity = 0;
    common->n_time_high_loop = 0;
    common->current_type = CD;
    common->first_time_base_set = false;
    common->eventnumcount = 0;
    common->pre_time_stamp = 0;
}

void finish_dpu(DPU_INTERFACE *interface, DPU_COMMON *cm)
{
    fcloseall();
    free_all(interface, cm);
}
