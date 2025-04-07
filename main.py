# intent-reconstruction/main.py
import argparse
from src.train import train
from src.evaluate import evaluate
from configs.model_configs import MODEL_CONFIGS_1B, MODEL_CONFIGS_3B

def get_all_model_names():
    """모든 사용 가능한 모델 이름을 반환"""
    model_names = []
    for config in [MODEL_CONFIGS_1B, MODEL_CONFIGS_3B]:
        model_names.extend([config[model_id].name for model_id in config])
    return model_names

def get_all_dataset_names():
    """모든 사용 가능한 데이터셋 이름을 반환"""
    return [
        "BlendATIS", "BlendSNIPS", "BlendCLINC150", "BlendBanking77",
        "MixATIS", "MixSNIPS", "MixCLINC150", "MixBanking77"
    ]

def main():
    parser = argparse.ArgumentParser(description="Intent Detection: Training and Evaluation")
    parser.add_argument("mode", choices=["train", "eval"], help="Mode to run: train or evaluate")
    
    # train 모드일 때만 model-size 인자 추가
    parser.add_argument("--model-size", choices=["1b", "3b"], help="Model size to train (1b or 3b)")
    
    # 모델 선택 인자 추가
    parser.add_argument("--model-name", choices=get_all_model_names(), 
                       help="Specific model to train/evaluate. If not specified, all models will be processed.")
    
    # 데이터셋 선택 인자 추가
    parser.add_argument("--dataset-name", choices=get_all_dataset_names(),
                       help="Specific dataset to use. If not specified, all datasets will be processed.")
    
    args = parser.parse_args()

    if args.mode == "train":
        if not args.model_size:
            parser.error("--model-size is required for training mode")
        train(model_size=args.model_size, model_name=args.model_name, dataset_name=args.dataset_name)
    elif args.mode == "eval":
        evaluate(model_name=args.model_name, dataset_name=args.dataset_name)

if __name__ == "__main__":
    main()
