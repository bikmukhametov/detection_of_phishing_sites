# Результаты отбора признаков

## Сравнение алгоритмов

| Алгоритм | Признаков | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|----------|-----------|--------|----|---------|
| Add-Del | 11 | 0.9457 | 0.9447 | 0.9585 | 0.9516 | 0.9876 |
| Genetic Algorithm | 25 | 0.9493 | 0.9437 | 0.9667 | 0.9550 | 0.9882 |
| Stochastic Search (SPA) | 7 | 0.9349 | 0.9158 | 0.9724 | 0.9432 | 0.9745 |

## Add-Del

**Статистика:**
- Количество отобранных признаков: 11
- Финальное значение Q (ошибка): 0.0543
- Итераций: 17

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
- Количество отобранных признаков: 25
- Финальное значение Q (ошибка): 0.0507
- Итераций: 35

**Метрики качества:**
- Accuracy: 0.9493
- Precision: 0.9437
- Recall: 0.9667
- F1 Score: 0.9550
- ROC-AUC: 0.9882

**Выбранные признаки (25):**
1. having_IP_Address
2. URL_Length
3. Shortining_Service
4. having_At_Symbol
5. double_slash_redirecting
6. Prefix_Suffix
7. having_Sub_Domain_phish
8. having_Sub_Domain_suspic
9. having_Sub_Domain_legit
10. SSLfinal_State
11. port
12. URL_of_Anchor_phish
13. URL_of_Anchor_suspic
14. URL_of_Anchor_legit
15. Links_in_tags
16. SFH
17. Redirect
18. on_mouseover
19. popUpWidnow
20. Iframe
21. DNSRecord
22. Page_Rank
23. Google_Index
24. Links_pointing_to_page
25. Statistical_report

---

## Stochastic Search (SPA)

**Статистика:**
- Количество отобранных признаков: 7
- Финальное значение Q (ошибка): 0.0651
- Итераций: 250

**Метрики качества:**
- Accuracy: 0.9349
- Precision: 0.9158
- Recall: 0.9724
- F1 Score: 0.9432
- ROC-AUC: 0.9745

**Выбранные признаки (7):**
1. having_At_Symbol
2. having_Sub_Domain_legit
3. SSLfinal_State
4. URL_of_Anchor_phish
5. Links_in_tags
6. Google_Index
7. Links_pointing_to_page

---

