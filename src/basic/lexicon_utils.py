def get_prefixes(entity, min_length=3, max_length=8):
    # computer science => ['comp sci', ...]
    words = entity.split()
    candidates = ['']
    candidates = []
    # TODO: Fix how prefixes is done
    for i in range(min_length, max_length):
        candidates.append(entity[:i])
    # for word in words:
    #     new_candidates = []
    #     for c in candidates:
    #         if len(word) < max_length:  # Keep word
    #             new_candidates.append(c + ' ' + word)
    #         else:
    #             for i in range(min_length, max_length):
    #                 new_candidates.append(c + '' + word[:i])
    #     candidates = new_candidates

    stripped = [c.strip() for c in candidates if c != entity]
    return stripped

def get_acronyms(entity):
    """
    Computes acronyms of entity, assuming entity has more than one token
    :param entity:
    :return:
    """
    words = entity.split()
    first_letters = ''.join([w[0] for w in words])
    acronyms = [first_letters]

    # Add acronyms using smaller number of first letters in phrase ('ucb' -> 'uc')
    for split in range(2, len(first_letters)):
        acronyms.append(first_letters[:split])

    return acronyms


alphabet = "abcdefghijklmnopqrstuvwxyz "
def get_edits(entity):
    if len(entity) < 3:
        return []
    edits = []
    for i in range(len(entity) + 1):
        prefix = entity[:i]
        # Insert
        suffix = entity[i:]
        for c in alphabet:
            new_word = prefix + c + suffix
            edits.append(new_word)

        if i == len(entity):
            continue

        # Delete
        suffix = entity[i+1:]
        new_word = prefix + suffix
        edits.append(new_word)

        # Substitute
        suffix = entity[i+1:]
        for c in alphabet:
            if c != entity[i]:
                new_word = prefix + c + suffix
                edits.append(new_word)

        # Transposition - swapping two letters
        for j in range(i+1, len(entity)):
            mid = entity[i+1:j]
            suffix = entity[j+1:]
            new_word = prefix + entity[j] + mid + entity[i] + suffix
            new_word = new_word.strip()
            if new_word != entity:
                edits.append(new_word)
    return edits


def get_morphological_variants(entity):
    """
    Computes stem of entity and creates morphological variants
    :param entity:
    :return:
    """
    results = []
    for suffix in ['ing']:
        if entity.endswith(suffix):
            base = entity[:-len(suffix)]
            results.append(base)
            # TODO: Can we get away with not hard-coding these variants?
            results.append(base + 'e')
            results.append(base + 's')
            results.append(base + 'er')
            results.append(base + 'ers')
    return results
