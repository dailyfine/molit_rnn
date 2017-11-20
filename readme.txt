
<< 구성 >>

1. src/ggi_predictor.py  -> 핵심 모듈. 학습 및 테스트
2. src/preprocessor.py -> 데이터 전처리기
3. src/data_generator.py -> 데이터 생성기
(processing module을 사용하여, 경매 데이터를 이용해서 
국토부 데이터를 순회하면서 디스크에 h5py 툴로 저장)
이미 데이터는 전처리되어 batch_data_h5py에 저장되어 있으므로 참고용으로만 사용.
best_model/ -> 제일 좋은 모델 저장 기록.

<< 개발 환경 및 언어 >>
Ubuntu 16.04.3 LTS (GNU/Linux 4.10.0-32-generic x86_64)
python3 & tensorflow-gpu

