# Карточка автомобиля — известна в момент приёма
CAR_CARD_FEATURES = [
    "mark", "model", "generation", "car_year", "mileage",
    "engine_cc", "body_type", "engine_type",
    "transmission", "drive_type", "color", "steering", "door_num",
    "keys_count", "on_guarantee",
    "car_category", "total_car_state",
]
# is_turbo, is_gas, is_pledge, seats_count, is_approved_by_rop исключены —
# на 100% NaN по всему датасету (0 непустых значений из 164 400 строк),
# см. data_diary.md, п. 8. Не утечка, а мёртвые поля: использовать нечего.

# Контекст обращения — известен в момент приёма
CONTEXT_FEATURES = [
    "transaction_group", "transaction_subgroup", "transaction",
    "is_mobile_purch", "is_comtrade", "branch_key",
]

# Рыночные агрегаты — привязаны к mark/model/car_year/месяцу, не к сделке
MARKET_FEATURES = [
    "market_avg_price", "market_ads_cnt", "market_avg_mileage",
    "market_median_price_w", "market_q1_price", "market_q3_price",
    "has_price_quantiles",
]

ALLOWED_FEATURES = CAR_CARD_FEATURES + CONTEXT_FEATURES + MARKET_FEATURES

# Мёртвые поля: 100% NaN на всех 164 400 строках evaluations.csv (проверено
# явно, не по выборке). Не запрещены по смыслу утечки, но бессмысленны как
# признаки. Держим отдельным списком, чтобы assert_no_leakage не считал их
# "неописанной" колонкой, и чтобы при следующей выгрузке данных было видно,
# заполнились ли они наконец.
DEAD_FEATURES = [
    "is_turbo", "is_gas", "is_pledge", "seats_count", "is_approved_by_rop",
]

# Явно запрещено — весь блок согласования цены и всё, что известно только
# после решения человека или после выкупа. Список ведём отдельно от
# ALLOWED_FEATURES (а не как "всё остальное"), чтобы при появлении новых
# колонок в датасете пайплайн падал явно, а не молча всё разрешал.
FORBIDDEN_FEATURES = [
    "purchaser_first_buy_price", "purchaser_first_rrc",
    "rgv_first_buy_price", "rgv_first_rrc",
    "first_pricer_buy_price", "first_pricer_rrc", "first_pricer_plan_rrc",
    "first_pricer_market_price",
    "purchaser_second_buy_price", "purchaser_second_rrc",
    "rgv_second_buy_price", "rgv_second_rrc", "rgv_plan_rrc",
    "second_pricer_buy_price", "second_pricer_rrc",
    "second_pricer_plan_rrc", "second_pricer_market_price",
    "confirmed_pricer_buy_price", "confirmed_pricer_rrc",
    "confirmed_rgv_buy_price", "confirmed_rgv_rrc",
    "rov_buy_price", "rov_rrc",
    "head_branch_buy_price", "head_branch_rrc", "head_branch_plan_rrc",
    "market_price", "didi_market_price", "dd_market_price",
    "dd_liquidity_value",
    "expenses_labor", "expenses_parts", "expenses_total",
    "client_expectation",  # до подтверждения момента фиксации в процессе
]


def assert_no_leakage(columns) -> None:
    """Падает, если во входных признаках есть что-то за пределами контракта."""
    cols = set(columns)
    leaked = cols & set(FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"Обнаружена утечка признаков: {sorted(leaked)}")
    dead = cols & set(DEAD_FEATURES)
    if dead:
        raise ValueError(
            f"Колонки на 100% NaN, использовать нечего: {sorted(dead)}. "
            f"Если в новой выгрузке они заполнились — проверьте руками и "
            f"перенесите в ALLOWED_FEATURES."
        )
    unknown = cols - set(ALLOWED_FEATURES)
    if unknown:
        raise ValueError(
            f"Колонки не описаны в контракте (ни allowed, ни forbidden, ни "
            f"dead): {sorted(unknown)}. Добавьте их явно в "
            f"feature_contract.py после проверки момента доступности."
        )
