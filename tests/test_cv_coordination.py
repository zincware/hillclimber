import hillclimber as pn

def test_coordination_cv_na_water_smiles(na_cl_water):
    """Test default FIRST strategy - only first groups."""
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMILESSelector(smiles="O")

    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
        group_reduction_1="all",
        group_reduction_2="com_per_group",
    )

    labels, lines = coordination_cv.to_plumed(na_cl_water)

    assert labels == ["cn"]
    assert lines ==[
"cn_g2_0: COM ATOMS=3,4,5",
"cn_g2_1: COM ATOMS=6,7,8",
"cn_g2_2: COM ATOMS=9,10,11",
"cn_g2_3: COM ATOMS=12,13,14",
"cn_g2_4: COM ATOMS=15,16,17",
"cn_g2_5: COM ATOMS=18,19,20",
"cn_g2_6: COM ATOMS=21,22,23",
"cn_g2_7: COM ATOMS=24,25,26",
"cn_g2_8: COM ATOMS=27,28,29",
"cn_g2_9: COM ATOMS=30,31,32",
"cn_g2_group: GROUP ATOMS=cn_g2_0,cn_g2_1,cn_g2_2,cn_g2_3,cn_g2_4,cn_g2_5,cn_g2_6,cn_g2_7,cn_g2_8,cn_g2_9",
"cn: COORDINATION GROUPA=1 GROUPB=cn_g2_group R_0=0.3 NN=6 D_0=0.0"
    ]
    # plumed is 1index, ase is 0index
    assert na_cl_water[0].symbol == "Na"
    assert list(na_cl_water[[2, 3, 4]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[5, 6, 7]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[8, 9, 10]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[11, 12, 13]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[14, 15, 16]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[17, 18, 19]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[20, 21, 22]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[23, 24, 25]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[26, 27, 28]].symbols) == ["O", "H", "H"]
    assert list(na_cl_water[[29, 30, 31]].symbols) == ["O", "H", "H"]

def test_coordination_cv_na_water_smarts_com_per_group(na_cl_water):
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMARTSSelector(pattern="[O]", hydrogens="exclude")

    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
        group_reduction_1="all",
        group_reduction_2="com_per_group",
    )

    labels, lines = coordination_cv.to_plumed(na_cl_water)
    assert labels == ["cn"]
    assert lines == [
        "cn_g2_0: COM ATOMS=3",
        "cn_g2_1: COM ATOMS=6",
        "cn_g2_2: COM ATOMS=9",
        "cn_g2_3: COM ATOMS=12",
        "cn_g2_4: COM ATOMS=15",
        "cn_g2_5: COM ATOMS=18",
        "cn_g2_6: COM ATOMS=21",
        "cn_g2_7: COM ATOMS=24",
        "cn_g2_8: COM ATOMS=27",
        "cn_g2_9: COM ATOMS=30",
        "cn_g2_group: GROUP ATOMS=cn_g2_0,cn_g2_1,cn_g2_2,cn_g2_3,cn_g2_4,cn_g2_5,cn_g2_6,cn_g2_7,cn_g2_8,cn_g2_9",
        "cn: COORDINATION GROUPA=1 GROUPB=cn_g2_group R_0=0.3 NN=6 D_0=0.0"
    ]
    assert na_cl_water[0].symbol == "Na"
    assert set(na_cl_water[[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]].symbols) == {"O"}


def test_coordination_cv_na_water_smarts_all(na_cl_water):
    x1_selector = pn.SMILESSelector(smiles="[Na+]")
    x2_selector = pn.SMARTSSelector(pattern="[O]", hydrogens="exclude")

    coordination_cv = pn.CoordinationNumberCV(
        x1=x1_selector,
        x2=x2_selector,
        prefix="cn",
        r_0=0.3,
        d_0=0.0,
        group_reduction_1="all",
        group_reduction_2="all",
        multi_group="first", # This should now work correctly
    )

    labels, lines = coordination_cv.to_plumed(na_cl_water)
    assert labels == ["cn"]
    # The `all` reduction should flatten the groups from the SMARTSSelector
    assert lines == [
        "cn: COORDINATION GROUPA=1 GROUPB=3,6,9,12,15,18,21,24,27,30 R_0=0.3 NN=6 D_0=0.0"
    ]
    assert na_cl_water[0].symbol == "Na"
    assert set(na_cl_water[[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]].symbols) == {"O"}
