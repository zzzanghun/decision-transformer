# 이진 분류 모델 최적화 가이드

## 개요
드론 장애물 맵과 드론 정보를 입력받아 안전 우선(0) 또는 효율 우선(1)을 예측하는 이진 분류 모델을 위한 최적화

## 모델 아키텍처 개선사항

### 1. MinkowskiEngine 인코더 강화
- **이전**: 4개 레이어 (1→32→64→128→256), BatchNorm 비활성화
- **최적화**: 4개 레이어 (1→64→128→256→512), BatchNorm 활성화
  - 채널 수 증가: 더 풍부한 특징 추출
  - BatchNorm 활성화: 학습 안정성 향상
  - Dropout 최소화: 마지막 레이어에만 0.1 적용 (모델 표현력 유지)

### 2. Global Pooling 개선
- **이전**: Average Pooling만 사용
- **최적화**: Average + Max Pooling 결합
  - 두 가지 관점에서 특징 추출
  - 더 강건한 표현 학습

### 3. 드론 정보 인코더 강화
- **이전**: 2층 (46→256→128)
- **최적화**: 3층 (46→256→384→256)
  - 더 깊은 네트워크로 표현력 향상
  - Dropout 최소화: 0.05만 적용

### 4. 분류기 개선
- **이전**: 5층, Sigmoid 출력
- **최적화**: 5층, Logit 출력
  - 더 깊은 분류 헤드 (512→384→256→128→1)
  - BCE with Logits 사용으로 수치 안정성 향상
  - Dropout 최소화: 초반 두 레이어에만 0.1 적용

### 5. Latent Dimension 증가
- **이전**: 128
- **최적화**: 256
  - 더 큰 표현 공간으로 복잡한 패턴 학습

## 학습 전략 최적화

### 1. 손실 함수
- **이전**: MSELoss (회귀 문제로 접근)
- **최적화**: BCEWithLogitsLoss (이진 분류에 최적)
  - 로짓 직접 사용으로 수치 안정성 확보
  - 분류 작업에 더 적합한 손실 함수

### 2. 옵티마이저
- **이전**: Adam with weight_decay=1e-4
- **최적화**: AdamW with weight_decay=1e-3
  - AdamW: 더 나은 정규화 효과
  - weight_decay 증가: 과적합 방지 강화
  - betas=(0.9, 0.999): 기본값 유지

### 3. 학습률 스케줄링
- **이전**: 고정 학습률 1e-4
- **최적화**:
  - 초기 학습률: 3e-4 (더 빠른 수렴)
  - Warmup: 10 에폭 동안 선형 증가
  - CosineAnnealingWarmRestarts (T_0=50, T_mult=2, eta_min=1e-6)
    - 주기적 학습률 재시작으로 local minima 탈출
    - 긴 에폭 학습에 적합

### 4. 배치 크기
- **이전**: 64
- **최적화**: 128
  - 더 안정적인 그래디언트 추정
  - 배치 정규화 효과 향상
  - 메모리가 허용하는 선에서 증가

### 5. 정규화
- **L1 정규화**: 비활성화 (이진 분류에는 불필요)
- **L2 정규화**: weight_decay=1e-3로 적용
- **Dropout**: 최소화 (0.05~0.1만 사용, 모델 표현력 유지)
  - 인코더: 마지막에만 0.1
  - 드론 인코더: 0.05
  - 분류기: 초반 두 레이어에만 0.1
- **Gradient Clipping**: max_norm=1.0 (학습 안정성)

### 6. 평가 메트릭
- **이전**: MSE, RMSE
- **최적화**: Loss + Accuracy
  - 이진 분류 정확도 추가
  - 모델 성능을 직관적으로 파악

## 추천 하이퍼파라미터

```python
# 모델
latent_dim = 256
dropout_rates = [0.05, 0.1]  # 최소화된 dropout

# 학습
batch_size = 128
learning_rate = 3e-4
weight_decay = 1e-3
warmup_epochs = 10

# 스케줄러
T_0 = 50  # 첫 번째 재시작까지 에폭
T_mult = 2  # 재시작 주기 배수
eta_min = 1e-6  # 최소 학습률

# 정규화
gradient_clip = 1.0
```

## 기대 효과

1. **수렴 속도**: 3배 더 빠른 초기 수렴 (warmup + 높은 초기 학습률)
2. **최종 성능**: 더 깊고 넓은 모델로 5-10% 정확도 향상 예상
3. **안정성**: BatchNorm + Gradient Clipping으로 학습 안정화
4. **모델 표현력**: 최소화된 Dropout으로 모델의 학습 능력 극대화
5. **일반화**: AdamW의 L2 정규화로 과적합 방지
6. **Long-term 학습**: Cosine Annealing으로 장기 학습 시 성능 향상

## 모니터링 권장사항

학습 중 다음 메트릭 추적:
- Train/Val Loss
- Train/Val Accuracy
- Learning Rate
- L1 Regularization (사용 시)

조기 종료 조건:
- Validation Accuracy가 일정 에폭 동안 개선되지 않을 때
- 또는 목표 정확도 달성 시

## 추가 개선 가능성

1. **데이터 증강**:
   - 장애물 맵 회전/반전
   - 드론 정보 노이즈 추가

2. **Focal Loss 고려**:
   - 클래스 불균형이 있다면 Focal Loss 사용

3. **Ensemble**:
   - 여러 시드로 학습한 모델 앙상블

4. **Mixed Precision Training**:
   - AMP (Automatic Mixed Precision) 사용으로 속도 향상

5. **Cross Validation**:
   - K-fold CV로 더 robust한 평가
