< Negotiation >
  if first time:
    proposal() # make counter offer
  if second time:
    persuade()
  if third time:
    compromise()
  if fourth time:
    final_call()

< Hardball >
  if first time:
    s = "you drive a hard bargain here!",
        "that is too low, I can't do that",
        "XYZ are worth 0 points to me, I can't take that"
    proposal()
  if second time:
    compromise()
  if third time:
    final_call()

<Clarify>
  if their_offer does not mention one or more items:
  assume they want 0 of those items
  repeat back what you think their offer is
  Watch for [yes, sure, yep, yup, ok]
  Example:
    THEM: i would like the basketball .
    assume their_offer = {'book':0, 'hat':0, 'ball':1}
    YOU: if you get the [basketball], does that mean i get everything else ?

during recieve, try to answer their questions
