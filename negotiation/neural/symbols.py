from collections import namedtuple

SpecialSymbols = namedtuple('SpecialSymbols',
        ['EOS', 'END_SUM', 'GO_S', 'GO_B', 'OFFER', 'QUIT', 'ACCEPT', 'REJECT', 'PAD', 'START_SLOT', 'END_SLOT', 'C_car', 'C_phone', 'C_housing', 'C_electronics', 'C_furniture', 'C_bike'])

markers = SpecialSymbols(EOS='</s>', END_SUM='</sum>', GO_S='<go-s>', GO_B='<go-b>', OFFER='<offer>', QUIT='<quit>', ACCEPT='<accept>', REJECT='<reject>', PAD='<pad>', START_SLOT='<slot>', END_SLOT='</slot>', C_car='<car>', C_phone='<phone>', C_housing='<housing>', C_electronics='<electronics>', C_furniture='<furniture>', C_bike='<bike>')
