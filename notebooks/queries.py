# pylint: disable=missing-docstring, C0103

def directors_count(db):
    # return the number of directors contained in the database
    # $CHALLENGIFY_BEGIN
    query = """
        SELECT COUNT(*)
        FROM directors
    """
    db.execute(query)
    count = db.fetchone()
    return count[0]
    # $CHALLENGIFY_END


def directors_list(db):
    # return the list of all the directors sorted in alphabetical order
    # $CHALLENGIFY_BEGIN
    query = """
        SELECT name
        FROM directors
        ORDER BY name
    """
    db.execute(query)
    directors = db.fetchall()
    return [director[0] for director in directors]
    # $CHALLENGIFY_END


def love_movies(db):
    # return the list of all movies which contain the word "love" in their title, sorted in alphabetical order
    # $CHALLENGIFY_BEGIN
    query = """
        SELECT title
        FROM movies
        WHERE title LIKE '%love%'
        ORDER BY title
    """
    db.execute(query)
    movies = db.fetchall()
    return [movie[0] for movie in movies]
    # $CHALLENGIFY_END


def directors_named_like_count(db, name):
    # return the number of directors which contain a given word in their name
    # $CHALLENGIFY_BEGIN
    query = """
        SELECT COUNT(*)
        FROM directors
        WHERE name LIKE ?
    """
    db.execute(query, (f"%{name}%",))
    count = db.fetchone()
    return count[0]
    # $CHALLENGIFY_END


def movies_longer_than(db, min_length):
    # return this list of all movies which are longer than a given duration, sorted in the alphabetical order
    # $CHALLENGIFY_BEGIN
    query = """
        SELECT title
        FROM movies
        WHERE minutes > ?
        ORDER BY title
    """
    db.execute(query, (min_length,))
    movies = db.fetchall()
    return [movie[0] for movie in movies]
    # $CHALLENGIFY_END
