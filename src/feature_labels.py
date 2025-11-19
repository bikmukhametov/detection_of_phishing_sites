"""
Helpers for translating feature names into Russian labels for visualization
and reporting outputs.
"""

from typing import Iterable, List

FEATURE_NAME_TRANSLATIONS = {
    "SSLfinal_State": "Статус_SSL",
    "URL_of_Anchor_phish": "Фишинговый_домен_автора",
    "URL_of_Anchor_legit": "Легитимный_домен_автора",
    "web_traffic_high": "Высокий_веб_трафик",
    "having_Sub_Domain_legit": "Число_поддоменов",
    "Prefix_Suffix": "Дефис_в_домене",
    "URL_of_Anchor_suspic": "Подозрительный_домен_автора",
    "Request_URL": "Внешние_ресурсы",
    "having_Sub_Domain_suspic": "Подозрительные_поддомены",
    "web_traffic_middle": "Средний_веб_трафик",
    "Links_in_tags": "Ссылки_в_тегах",
    "Domain_registeration_length": "Срок_регистрации",
    "web_traffic_low": "Низкий_веб_трафик",
    "SFH": "Обработчик_форм",
    "having_Sub_Domain_phish": "Фишинговые_поддомены",
    "Google_Index": "Индекс_Google",
    "age_of_domain": "Возраст_домена",
    "Page_Rank": "Оценка PageRank",
    "having_IP_Address": "IP_вместо_домена",
    "Statistical_report": "Фishing_в_отчетах",
    "DNSRecord": "DNS_записи",
    "Shortining_Service": "Сервис_сокращения",
    "Abnormal_URL": "Аномальный_URL",
    "Links_pointing_to_page": "Внешние_ссылки",
    "having_At_Symbol": "Символ_@",
    "URL_Length": "Длина_URL",
    "on_mouseover": "Эффект_наведения",
    "HTTPS_token": "HTTPS_в_домене",
    "double_slash_redirecting": "Двойной_слеш",
    "port": "Подозрительный_порт",
    "Redirect": "Число_редиректов",
    "Submitting_to_email": "Отправка_на_почту",
    "RightClick": "Правый_клик",
    "Iframe": "Невидимый_iframe",
    "Favicon": "Сторонняя_иконка",
    "popUpWidnow": "Всплывающие_окна",
}


def translate_feature_name(name: str) -> str:
    """Return the Russian label for a feature if available."""
    return FEATURE_NAME_TRANSLATIONS.get(name, name)


def translate_feature_sequence(names: Iterable[str]) -> List[str]:
    """Translate an iterable of feature names preserving order."""
    return [translate_feature_name(name) for name in names]

