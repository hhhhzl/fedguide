# Bandit2D 对比验证结果

**实验配置**: 4 clients, 60 rounds, seed 0, n_steps 200  
**运行时间**: 2026-03-13

---

## 完整对比表

| 方法 | train/return (final) | train/return (best) | train/return (AUC) | eval/return (final) | eval/return (best) | eval/return (AUC) |
|------|---------------------|---------------------|--------------------|--------------------|--------------------|-------------------|
| **FedGuide** | 22.25 | 22.56 | 19.92 | **0.096** | 0.28 | **0.075** |
| **FedKL**    | 22.41 | 25.46 | 21.19 | 0.002 | 0.002 | 0.0002 |
| **FedAvg**   | **22.80** | **27.06** | **22.34** | 0.046 | **0.43** | 0.066 |

---

## 验证结论

### train/return：同事说法成立 ✓

FedGuide 的 train/return **确实低于** FedKL 和 FedAvg：

- **Final**: FedGuide (22.25) < FedKL (22.41) < FedAvg (22.80)
- **Best**: FedGuide (22.56) < FedKL (25.46) < FedAvg (27.06)
- **AUC**: FedGuide (19.92) < FedKL (21.19) < FedAvg (22.34)

### eval/return：同事说法不成立 ✗

FedGuide 的 eval/return **优于** FedKL，与 FedAvg 相当或更好：

- **Final**: FedGuide (0.096) > FedAvg (0.046) >> FedKL (0.002)
- **Best**: FedAvg (0.43) > FedGuide (0.28) >> FedKL (0.002)
- **AUC**: FedGuide (0.075) > FedAvg (0.066) >> FedKL (0.0002)

---

## 总结

| 指标 | 同事说法 | 实际验证 |
|------|----------|----------|
| train/return | FedGuide < FedKL/FedAvg | ✓ **成立** |
| eval/return  | FedGuide < FedKL/FedAvg | ✗ **不成立**（FedGuide 优于 FedKL，与 FedAvg 相当） |

**结论**：train/return 上 FedGuide 确实更差；eval/return 上 FedGuide 明显优于 FedKL，与 FedAvg 接近或略优。
