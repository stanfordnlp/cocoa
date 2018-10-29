#from collections import namedtuple
#
#SpecialSymbols = namedtuple('SpecialSymbols',
#        ['EOS', 'END_SUM', 'GO_S', 'GO_B', 'OFFER', 'QUIT', 'ACCEPT', 'REJECT', 'PAD', 'C_car', 'C_phone', 'C_housing', 'C_electronics', 'C_furniture', 'C_bike'])
#
#markers = SpecialSymbols(EOS='</s>', END_SUM='</sum>', GO_S='<go-s>', GO_B='<go-b>', OFFER='<offer>', QUIT='<quit>', ACCEPT='<accept>', REJECT='<reject>', PAD='<pad>', C_car='<car>', C_phone='<phone>', C_housing='<housing>', C_electronics='<electronics>', C_furniture='<furniture>', C_bike='<bike>')

from cocoa.neural.symbols import Marker as BaseMarker

class Marker(BaseMarker):
    # Sequence
    END_SUM = '</sum>'
    GO_S = '<go-s>'
    GO_B = '<go-b>'

    # Actions
    OFFER = '<offer>'
    QUIT = '<quit>'
    ACCEPT = '<accept>'
    REJECT = '<reject>'

    # Categories
    C_car = '<car>'
    C_phone = '<phone>'
    C_housing = '<housing>'
    C_electronics = '<electronics>'
    C_furniture = '<furniture>'
    C_bike = '<bike>'

markers = Marker

category_markers = [markers.C_car, markers.C_phone, markers.C_housing, markers.C_electronics, markers.C_furniture, markers.C_bike]

action_markers = [markers.ACCEPT, markers.REJECT, markers.OFFER, markers.QUIT]

sequence_markers = [markers.EOS, markers.GO_S, markers.GO_B, markers.PAD]

