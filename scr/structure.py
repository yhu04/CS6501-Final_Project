


class Review:
    def __init__(self, review, Vocab, VocabDict):
        self.business_id = review.get("business_id")
        self.user_id = review.get("user_id")
        self.stars = review.get("stars")
        self.text = review.get("text")
        self.date = review.get("date")
        self.votes = review.get("votes")


# {
#     'type': 'review',
#     'business_id': (encrypted business id),
#     'user_id': (encrypted user id),
#     'stars': (star rating, rounded to half-stars),
#     'text': (review text),
#     'date': (date, formatted like '2012-03-14'),
#     'votes': {(vote type): (count)},
# }

class Business:
    def __init__(self, business, Vocab, VocabDict):
        self.business_id = business.get("business_id")
        self.name = business.get("name")
        self.neighborhoods = business.get("neighborhoods")
        self.full_address = business.get("full_address")
        self.city = business.get("city")
        self.state = business.get("state")
        self.latitude = business.get("latitude")
        self.longitude = business.get("longitude")
        self.stars = business.get("stars")
        self.review_count = business.get("review_count")
        self.categories = business.get("categories")
        self.open = business.get("open")
        self.hours = business.get("hours")
        self.attributes = business.get("attributes")




# {
#     'type': 'business',
#     'business_id': (encrypted business id),
#     'name': (business name),
#     'neighborhoods': [(hood names)],
#     'full_address': (localized address),
#     'city': (city),
#     'state': (state),
#     'latitude': latitude,
#     'longitude': longitude,
#     'stars': (star rating, rounded to half-stars),
#     'review_count': review count,
#     'categories': [(localized category names)]
#     'open': True / False (corresponds to closed, not business hours),
#     'hours': {
#         (day_of_week): {
#             'open': (HH:MM),
#             'close': (HH:MM)
#         },
#         ...
#     },
#     'attributes': {
#         (attribute_name): (attribute_value),
#         ...
#     },
# }

class User:
    def __init__(self, user, Vocab, VocabDict):
        self.user_id = user.get("user_id")
        self.name = user.get("name")
        self.review_count = user.get("review_count")
        self.average_stars = user.get("average_stars")
        self.votes = user.get("votes")
        self.friends = user.get("friends")
        self.elite = user.get("elite")
        self.yelping_since = user.get("yelping_since")
        self.compliments = user.get("compliments")
        self.fans = user.get("fans")



# {
#     'type': 'user',
#     'user_id': (encrypted user id),
#     'name': (first name),
#     'review_count': (review count),
#     'average_stars': (floating point average, like 4.31),
#     'votes': {(vote type): (count)},
#     'friends': [(friend user_ids)],
#     'elite': [(years_elite)],
#     'yelping_since': (date, formatted like '2012-03'),
#     'compliments': {
#         (compliment_type): (num_compliments_of_this_type),
#         ...
#     },
#     'fans': (num_fans),
# }
