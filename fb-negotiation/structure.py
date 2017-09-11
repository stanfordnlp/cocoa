<Start>
  if their message contains an offer:
    <Receive> or <Clarify>
  elif their message has no offer:
    <Send>
    if p < 0.5:
      introduce()     # talk a bit, to hear their side first
    else:
      init_propose()  # make a light proposal, to get the ball rolling

<Receive>
  their_offer is below bottomline: # they are playing hardball
    hardball()
  their_offer is above bottomline but below good_deal:  # we negotiate
    negotiate()
  their_offer is good_deal:
    agree()
  Still do not make an offer after init_propose:
    proposal()


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

At any time, when <Receive>
  if good_deal: (aka >= 8 points)
    agree()
  if their_offer matches my_proposal:
    agree()
    <Select>
  if they agree to my_proposal:
    <Select>

def final_call():
  # If they are only offering 0 or 1 points, then
  # might as well reject since "No Deal" does not cause negative reward
  if 2 points or greater:
    agree()
  else:
    reject()

good_deal is >= 8 points
bottomline is >= point total from
  A = get one item of highest value
  B = get two items of lowest value
  C = 5 points
  pick min(A, B, C)
final_call is >= 2 points

def compromise():
  A = their_offer - 1 item of highest value
  B = my_proposal + 1 item of lowest value
  return min(A, B)

def agree():
  "ok deal, thanks!"
  "yes, that sounds good"
  "perfect, sounds like we have a deal"
  "ok, it's a deal"
