from collections import namedtuple

SpecialSymbols = namedtuple('SpecialSymbols',
        ['EOS', 'GO_S', 'GO_B', 'OFFER', 'QUIT', 'ACCEPT', 'REJECT', 'PAD', 'START_SLOT', 'END_SLOT', 'C_car', 'C_phone', 'C_housing', 'C_electronics', 'C_furniture', 'C_bike'])

markers = SpecialSymbols(EOS='</s>', GO_S='<go-s>', GO_B='<go-b>', OFFER='<offer>', QUIT='<quit>', ACCEPT='<accept>', REJECT='<reject>', PAD='<pad>', START_SLOT='<slot>', END_SLOT='</slot>', C_car='<car>', C_phone='<phone>', C_housing='<housing>', C_electronics='<electronics>', C_furniture='<furniture>', C_bike='<bike>')

category_markers = [markers.C_car, markers.C_phone, markers.C_housing, markers.C_electronics, markers.C_furniture, markers.C_bike]
