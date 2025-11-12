# Результаты отбора признаков

## Сравнение алгоритмов

| Алгоритм | Признаков | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|----------|-----------|--------|----|---------|
| Add-Del | 11 | 0.9457 | 0.9447 | 0.9585 | 0.9516 | 0.9876 |
| Genetic Algorithm | 26 | 0.9466 | 0.9413 | 0.9642 | 0.9526 | 0.9880 |
| Stochastic Search (SPA) | 19 | 0.9448 | 0.9301 | 0.9740 | 0.9515 | 0.9866 |

## Add-Del

**Статистика:**
- Количество отобранных признаков: 11
- Финальное значение Q (ошибка): 0.0543
- Итераций: 24

**Метрики качества:**
- Accuracy: 0.9457
- Precision: 0.9447
- Recall: 0.9585
- F1 Score: 0.9516
- ROC-AUC: 0.9876

**Выбранные признаки (11):**
1. having_IP_Address
2. URL_Length
3. Prefix_Suffix
4. having_Sub_Domain_legit
5. SSLfinal_State
6. Domain_registeration_length
7. URL_of_Anchor_phish
8. URL_of_Anchor_suspic
9. Links_in_tags
10. SFH
11. Google_Index

---

## Genetic Algorithm

**Статистика:**
- Количество отобранных признаков: 26
- Финальное значение Q (ошибка): 0.0534
- Итераций: 30

**Метрики качества:**
- Accuracy: 0.9466
- Precision: 0.9413
- Recall: 0.9642
- F1 Score: 0.9526
- ROC-AUC: 0.9880

**Выбранные признаки (26):**
1. having_IP_Address
2. URL_Length
3. double_slash_redirecting
4. Prefix_Suffix
5. having_Sub_Domain_phish
6. having_Sub_Domain_suspic
7. having_Sub_Domain_legit
8. SSLfinal_State
9. Domain_registeration_length
10. Favicon
11. port
12. URL_of_Anchor_phish
13. URL_of_Anchor_legit
14. Links_in_tags
15. SFH
16. Submitting_to_email
17. Abnormal_URL
18. Redirect
19. RightClick
20. popUpWidnow
21. Iframe
22. DNSRecord
23. Page_Rank
24. Google_Index
25. Links_pointing_to_page
26. Statistical_report

---

## Stochastic Search (SPA)

**Статистика:**
- Количество отобранных признаков: 19
- Финальное значение Q (ошибка): 0.0552
- Итераций: 540

**Метрики качества:**
- Accuracy: 0.9448
- Precision: 0.9301
- Recall: 0.9740
- F1 Score: 0.9515
- ROC-AUC: 0.9866

**Выбранные признаки (19):**
1. having_IP_Address
2. Shortining_Service
3. having_At_Symbol
4. double_slash_redirecting
5. Prefix_Suffix
6. having_Sub_Domain_suspic
7. having_Sub_Domain_legit
8. SSLfinal_State
9. port
10. HTTPS_token
11. URL_of_Anchor_phish
12. URL_of_Anchor_legit
13. Links_in_tags
14. Submitting_to_email
15. Abnormal_URL
16. Iframe
17. DNSRecord
18. Google_Index
19. Links_pointing_to_page

---

