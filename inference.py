import logging
import soundfile
from inference import infer_tool
from inference.infer_tool import Svc
logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth")
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json")
    parser.add_argument('-cl', '--clip', type=float, default=0)
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=None)
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0])
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['buyizi'])

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-40)
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4)
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5)
    parser.add_argument('-wf', '--wav_format', type=str, default='flac')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float)
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0)


    args = parser.parse_args()

    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    shallow_diffusion = args.shallow_diffusion
    use_spk_mix = args.use_spk_mix
    second_encoding = args.second_encoding
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    if cluster_infer_ratio != 0:
        if args.cluster_model_path == "":
            if args.feature_retrieval:  # 若指定了占比但没有指定模型路径，则按是否使用特征检索分配默认的模型路径
                args.cluster_model_path = "logs/44k/feature_and_index.pkl"
            else:
                args.cluster_model_path = "logs/44k/kmeans_10000.pt"
    else:  # 若未指定占比，则无论是否指定模型路径，都将其置空以避免之后的模型加载
        args.cluster_model_path = ""

    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    use_spk_mix,
                    args.feature_retrieval)
    
    infer_tool.mkdir(["raw", "results"])
    
    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step":k_step,
                "use_spk_mix":use_spk_mix,
                "second_encoding":second_encoding,
                "loudness_envelope_adjustment":loudness_envelope_adjustment
            }
            audio = svc_model.slice_inference(**kwarg)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion :
                isdiffusion = "sovdiff"
            if only_diffusion :
                isdiffusion = "diff"
            if use_spk_mix:
                spk = "spk_mix"
            res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
