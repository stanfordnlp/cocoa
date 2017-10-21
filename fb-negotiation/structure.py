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


Decoding

g = 2 0 3 1 1 7

2 books available
Worth 0 points to you
3 hats available
Worth 1 point each
1 ball available
Worth 7 massive points



# 1) Build out Scenarios - probably hardcoded
# 2) Draft schema
# 3) Figure out the KB for each agent (within the Session)
# 4) At turn 10, give best offer, if not taken, then quit with value of 0

--Notes--
For Craigslist, the buyer always speaks first
The description in the KB for the seller is always exactly twice the length as it is for the buyer, why is that?