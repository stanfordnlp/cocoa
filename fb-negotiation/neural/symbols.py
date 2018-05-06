from cocoa.neural.symbols import Marker as BaseMarker

# Facebook Negotiation
class Marker(BaseMarker):
    # Sequence
    GO = '<go>'

    # Actions
    SELECT = '<select>'
    # OFFER = '<offer>'
    # ACCEPT = '<accept>'
    # REJECT = '<reject>'
    QUIT = '<quit>'

    # Items
    C_book = '<book>'
    C_hat = '<hat>'
    C_ball = '<ball>'
    '''
    Items are currently represented by Entities,
    These markers are created as just a precaution
    '''

markers = Marker
sequence_markers = [markers.EOS, markers.GO, markers.PAD]
action_markers = [markers.SELECT, markers.QUIT]
item_markers = [markers.C_book, markers.C_hat, markers.C_ball]

