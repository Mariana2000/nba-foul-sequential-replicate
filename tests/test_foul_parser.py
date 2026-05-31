from whistle_balance.foul_parser import is_foul_description, classify_foul

def test_is_foul_description():
    assert is_foul_description("Personal foul by Player X")
    assert not is_foul_description("Player X makes 2-pt shot")

def test_classify_foul():
    result = classify_foul("Shooting foul by Player X")
    assert result["shooting_foul"] == 1
    assert result["foul_type"] == "shooting"
    result2 = classify_foul("Turner S.FOUL (P1.T2)", sub_type="Shooting")
    assert result2["shooting_foul"] == 1
