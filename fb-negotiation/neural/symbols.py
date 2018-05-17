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

markers = Marker
sequence_markers = [markers.EOS, markers.GO, markers.PAD]
action_markers = [markers.SELECT, markers.QUIT]
