"""Generate original synthetic examples; these are not real TMDB records."""
import csv
import json
from pathlib import Path

GROUPS = {
    "Science Fiction": ("Orbit", "space astronaut planet galaxy spacecraft mission distant stars"),
    "Crime": ("Case", "detective crime police mystery evidence investigation suspect city"),
    "Romance": ("Hearts", "love romance wedding relationship couple family summer"),
}


def generate(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "title", "overview", "genres"])
        writer.writeheader()
        for group, (genre, text) in enumerate(GROUPS.items()):
            title, overview = text
            for number in range(6):
                writer.writerow({"id": 90000001 + group * 6 + number,
                                 "title": f"{title} {number + 1}",
                                 "overview": f"{overview} adventure chapter {number + 1}",
                                 "genres": json.dumps([genre])})


if __name__ == "__main__":
    generate("data/demo_movies.csv")
