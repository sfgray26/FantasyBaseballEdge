"""
Player Big Board — 2026 MLB Draft Rankings

200+ players with Steamer/ZiPS consensus projections tuned for
Treemendous league categories (18 cats, H2H One Win).

Each player dict:
  id, name, team, positions, type (batter/pitcher),
  tier (1-10), rank (overall), adp (approximate Yahoo ADP),
  proj (dict of projected stats), z_score, cat_scores

Category directions:
  Batter K  → negative (lower is better for your team)
  Pitcher L  → negative
  Pitcher HR → negative
  ERA, WHIP  → negative (lower is better)
  All others → positive

Run this module standalone to see rankings:
  python -m backend.fantasy_baseball.player_board
"""

import statistics
from typing import Optional


# ---------------------------------------------------------------------------
# Raw player data (2026 Steamer/ZiPS consensus)
# Format: (name, team, positions, type, tier, adp,
#          pa/ip, r/w, h/l, hr/sv, rbi/bs, k_bat/qs, tb/k_pit, avg/era, ops/whip, nsb/k9, nsv/hr_pit)
# ---------------------------------------------------------------------------

_BATTER_RAW = [
    # Tier 1 — Elite multi-cat (pick 1-12)
    ("Ronald Acuna Jr.", "ATL", ["LF", "CF", "RF", "OF"], 1, 1.0,
     680, 115, 190, 37, 95, 145, 345, 0.306, 0.995, 58, 0),
    ("Juan Soto",        "NYM", ["LF", "RF", "OF"],        1, 2.0,
     680, 105, 172, 38, 100, 128, 330, 0.288, 0.977, 7, 0),
    ("Shohei Ohtani",   "LAD", ["DH", "OF"],               1, 3.0,
     650, 100, 165, 45, 115, 150, 365, 0.282, 0.988, 15, 0),
    ("Mookie Betts",    "LAD", ["SS", "RF", "OF"],         1, 4.0,
     650, 110, 180, 32, 92, 130, 318, 0.295, 0.940, 14, 0),
    ("Yordan Alvarez",  "HOU", ["DH", "LF", "OF"],         1, 5.0,
     620, 95, 165, 38, 110, 138, 342, 0.295, 0.975, 3, 0),
    ("Freddie Freeman", "LAD", ["1B"],                      1, 6.0,
     660, 100, 192, 28, 102, 118, 312, 0.305, 0.916, 10, 0),
    ("Bobby Witt Jr.",  "KC",  ["SS"],                      1, 7.0,
     660, 108, 188, 28, 96, 125, 302, 0.295, 0.895, 42, 0),
    ("Corey Seager",    "TEX", ["SS"],                      1, 8.0,
     580, 92, 166, 32, 98, 108, 300, 0.298, 0.928, 5, 0),
    ("Gunnar Henderson","BAL", ["SS", "3B"],                1, 9.0,
     640, 102, 172, 34, 98, 162, 315, 0.280, 0.902, 18, 0),
    ("José Ramirez",    "CLE", ["3B"],                      1, 10.0,
     650, 100, 175, 28, 102, 82, 298, 0.285, 0.888, 26, 0),
    ("Julio Rodriguez", "SEA", ["CF", "OF"],                1, 11.0,
     650, 95, 175, 30, 90, 158, 298, 0.285, 0.872, 30, 0),
    ("Francisco Lindor","NYM", ["SS"],                      1, 12.0,
     640, 95, 166, 30, 97, 140, 298, 0.278, 0.869, 22, 0),

    # Tier 2 — Strong starters (pick 13-36)
    ("Bryce Harper",    "PHI", ["1B", "RF", "OF"],         2, 15.0,
     590, 92, 162, 32, 97, 118, 300, 0.288, 0.938, 8, 0),
    ("Elly De La Cruz", "CIN", ["SS", "3B"],                2, 16.0,
     630, 97, 162, 26, 87, 198, 278, 0.265, 0.835, 52, 0),
    ("Trea Turner",     "PHI", ["SS"],                      2, 17.0,
     620, 96, 176, 22, 80, 128, 285, 0.295, 0.862, 30, 0),
    ("Marcus Semien",   "TEX", ["2B", "SS"],                2, 18.0,
     660, 100, 170, 25, 88, 132, 285, 0.270, 0.842, 12, 0),
    ("Pete Alonso",     "NYM", ["1B"],                      2, 21.0,
     620, 82, 150, 40, 108, 152, 308, 0.256, 0.876, 2, 0),
    ("Paul Goldschmidt","STL", ["1B"],                      2, 22.0,
     600, 88, 162, 26, 92, 128, 280, 0.282, 0.880, 8, 0),
    ("Adley Rutschman", "BAL", ["C"],                       2, 24.0,
     580, 80, 156, 20, 82, 95, 260, 0.278, 0.848, 5, 0),
    ("Jazz Chisholm",   "NYY", ["LF", "2B", "OF"],         2, 28.0,
     560, 84, 148, 24, 80, 152, 255, 0.266, 0.822, 24, 0),
    ("William Contreras","MIL",["C", "1B"],                 2, 30.0,
     570, 78, 152, 24, 82, 102, 258, 0.278, 0.848, 5, 0),
    ("CJ Abrams",       "WSH", ["SS"],                      2, 32.0,
     640, 98, 180, 22, 78, 148, 280, 0.290, 0.852, 36, 0),
    ("Nolan Arenado",   "STL", ["3B"],                      2, 35.0,
     600, 85, 162, 28, 98, 112, 285, 0.278, 0.862, 4, 0),
    ("Rafael Devers",   "BOS", ["3B", "DH"],                2, 36.0,
     610, 88, 162, 28, 98, 138, 285, 0.272, 0.868, 4, 0),

    # Tier 3 — Solid contributors (pick 37-72)
    ("Kyle Tucker",     "CHC", ["RF", "OF"],                3, 40.0,
     580, 85, 155, 28, 88, 142, 275, 0.278, 0.885, 15, 0),
    ("Willy Adames",    "SF",  ["SS"],                      3, 42.0,
     600, 85, 155, 28, 90, 155, 270, 0.268, 0.848, 12, 0),
    ("Matt Olson",      "ATL", ["1B"],                      3, 44.0,
     620, 88, 155, 36, 105, 168, 305, 0.252, 0.870, 2, 0),
    ("Yordan Alvarez",  "HOU", ["DH"],                      3, 45.0,
     610, 90, 158, 35, 102, 142, 325, 0.278, 0.942, 3, 0),
    ("Cody Bellinger",  "CHC", ["1B", "CF", "OF"],          3, 48.0,
     580, 82, 158, 22, 80, 128, 265, 0.278, 0.858, 12, 0),
    ("Brent Rooker",    "OAK", ["LF", "DH", "OF"],          3, 50.0,
     560, 80, 140, 32, 92, 165, 272, 0.265, 0.878, 4, 0),
    ("Christopher Morel","CHC",["3B", "2B"],                3, 52.0,
     560, 78, 142, 24, 80, 162, 250, 0.265, 0.838, 18, 0),
    ("Eugenio Suarez",  "ARI", ["3B"],                      3, 55.0,
     570, 78, 138, 28, 88, 172, 255, 0.252, 0.838, 4, 0),
    ("Manny Machado",   "SD",  ["3B", "SS"],                3, 58.0,
     590, 82, 155, 24, 88, 118, 262, 0.272, 0.842, 6, 0),
    ("Jarred Kelenic",  "ATL", ["LF", "RF", "OF"],          3, 60.0,
     550, 80, 145, 24, 78, 152, 252, 0.272, 0.852, 18, 0),
    ("Seiya Suzuki",    "CHC", ["RF", "OF"],                3, 62.0,
     560, 78, 150, 22, 78, 128, 255, 0.278, 0.855, 8, 0),
    ("Wyatt Langford",  "TEX", ["LF", "CF", "RF", "OF"],   3, 64.0,
     590, 85, 158, 22, 82, 145, 262, 0.278, 0.858, 20, 0),
    ("Riley Greene",    "DET", ["CF", "OF"],                3, 65.0,
     580, 82, 155, 22, 80, 148, 258, 0.275, 0.850, 14, 0),
    ("Maikel Garcia",   "KC",  ["3B", "SS"],                3, 68.0,
     570, 78, 158, 12, 65, 98, 218, 0.290, 0.808, 28, 0),
    ("Ke'Bryan Hayes",  "PIT", ["3B"],                      3, 70.0,
     530, 70, 140, 14, 62, 108, 215, 0.278, 0.808, 16, 0),

    # Tier 4 — Good depth (pick 73-108)
    ("J.T. Realmuto",   "PHI", ["C"],                       4, 75.0,
     520, 68, 132, 18, 72, 108, 228, 0.268, 0.828, 8, 0),
    ("Austin Wells",    "NYY", ["C"],                       4, 80.0,
     510, 65, 128, 20, 70, 118, 232, 0.265, 0.822, 4, 0),
    ("Connor Wong",     "BOS", ["C"],                       4, 85.0,
     490, 62, 122, 18, 65, 122, 218, 0.262, 0.808, 8, 0),
    ("Salvador Perez",  "KC",  ["C", "DH"],                 4, 88.0,
     550, 68, 140, 22, 80, 112, 242, 0.268, 0.822, 2, 0),
    ("Jake Cronenworth","SD",  ["1B", "2B"],                4, 90.0,
     560, 75, 148, 16, 72, 108, 235, 0.275, 0.818, 10, 0),
    ("Zach McKinstry",  "DET", ["2B", "3B", "LF"],          4, 92.0,
     520, 68, 132, 16, 65, 122, 218, 0.265, 0.808, 14, 0),
    ("Jonathan India",  "CIN", ["2B"],                      4, 95.0,
     570, 78, 148, 18, 72, 135, 240, 0.272, 0.832, 15, 0),
    ("Ezequiel Tovar",  "COL", ["SS"],                      4, 98.0,
     560, 75, 148, 20, 75, 128, 248, 0.272, 0.835, 16, 0),
    ("Ian Happ",        "CHC", ["LF", "RF", "OF"],          4, 100.0,
     560, 78, 142, 20, 75, 138, 248, 0.265, 0.838, 8, 0),
    ("Anthony Santander","TOR",["LF", "RF", "DH", "OF"],   4, 102.0,
     580, 80, 148, 28, 88, 145, 268, 0.262, 0.848, 4, 0),
    ("Teoscar Hernandez","LAD",["LF", "RF", "OF"],          4, 105.0,
     570, 78, 145, 28, 88, 158, 262, 0.262, 0.842, 8, 0),
    ("Jackson Merrill",  "SD", ["CF", "OF"],                4, 108.0,
     570, 78, 152, 22, 78, 118, 252, 0.275, 0.845, 14, 0),
    ("Michael Harris",  "ATL", ["CF", "OF"],                4, 110.0,
     580, 82, 155, 22, 78, 135, 252, 0.272, 0.838, 22, 0),

    # Tier 5 — Solid bench/streaming (pick 109-144)
    ("Nolan Jones",     "COL", ["LF", "RF", "3B", "OF"],   5, 115.0,
     530, 72, 130, 20, 72, 148, 228, 0.262, 0.838, 14, 0),
    ("Oswaldo Cabrera", "NYY", ["2B", "SS", "3B", "LF"],   5, 118.0,
     520, 68, 132, 16, 65, 128, 222, 0.262, 0.808, 12, 0),
    ("Spencer Steer",   "CIN", ["1B", "2B", "3B", "OF"],   5, 120.0,
     580, 78, 148, 22, 82, 130, 252, 0.265, 0.802, 12, 0),
    ("David Peralta",   "LAD", ["LF", "RF", "OF"],          5, 122.0,
     480, 62, 128, 14, 62, 92, 202, 0.278, 0.818, 5, 0),
    ("Jorge Mateo",     "BAL", ["SS", "2B", "CF"],          5, 125.0,
     490, 68, 122, 10, 48, 138, 178, 0.258, 0.738, 38, 0),
    ("TJ Friedl",       "CIN", ["CF", "LF", "OF"],          5, 128.0,
     520, 72, 132, 14, 58, 118, 205, 0.268, 0.798, 28, 0),
    ("Esteury Ruiz",    "MIL", ["CF", "OF"],                5, 130.0,
     510, 68, 128, 8, 48, 122, 175, 0.260, 0.735, 52, 0),
    ("Jose Siri",       "TB",  ["CF", "LF", "OF"],          5, 132.0,
     480, 65, 115, 16, 55, 168, 198, 0.248, 0.768, 30, 0),
    ("Jahmai Jones",    "SEA", ["2B"],                      5, 135.0,
     510, 68, 128, 12, 58, 118, 198, 0.262, 0.782, 18, 0),
    ("Davis Schneider", "TOR", ["2B", "3B", "LF"],          5, 138.0,
     510, 68, 128, 18, 65, 138, 222, 0.262, 0.820, 10, 0),
    ("Nick Gonzales",   "PIT", ["2B"],                      5, 140.0,
     520, 70, 132, 16, 65, 125, 218, 0.265, 0.810, 14, 0),
    ("Tyler Stephenson","CIN", ["C"],                       5, 142.0,
     480, 58, 122, 14, 62, 105, 210, 0.265, 0.808, 2, 0),

    # ── MISSED TOP-TIER ADDITIONS ──────────────────────────────────────────
    # Aaron Judge — massive oversight in original board
    ("Aaron Judge",     "NYY", ["RF", "CF", "OF"],          1, 4.5,
     540, 95, 145, 45, 108, 175, 340, 0.278, 0.978, 6, 0),
    # Jarren Duran — breakout bat, speed elite
    ("Jarren Duran",    "BOS", ["CF", "LF", "OF"],          3, 58.0,
     580, 88, 165, 18, 72, 138, 258, 0.292, 0.848, 28, 0),
    # Luis Arraez — AVG/OBP category specialist
    ("Luis Arraez",     "SD",  ["1B", "2B"],                4, 112.0,
     550, 68, 178, 5, 48, 48, 202, 0.332, 0.798, 6, 0),
    # Daulton Varsho — multi-pos C/CF value
    ("Daulton Varsho",  "TOR", ["C", "CF", "OF"],           4, 116.0,
     500, 65, 122, 18, 62, 148, 218, 0.252, 0.782, 12, 0),

    # Tier 5 cont — ADP 145-200
    ("Cal Raleigh",     "SEA", ["C", "1B"],                 5, 148.0,
     520, 68, 130, 25, 72, 148, 238, 0.262, 0.828, 3, 0),
    ("Nathaniel Lowe",  "TEX", ["1B"],                      5, 152.0,
     570, 72, 158, 18, 72, 122, 252, 0.282, 0.818, 4, 0),
    ("Ha-Seong Kim",    "SD",  ["SS", "2B", "3B"],          5, 155.0,
     540, 68, 138, 14, 58, 118, 215, 0.265, 0.782, 22, 0),
    ("Corbin Carroll",  "ARI", ["CF", "LF", "OF"],          5, 158.0,
     560, 78, 148, 18, 68, 148, 252, 0.275, 0.828, 28, 0),
    ("Randy Arozarena", "SEA", ["LF", "RF", "OF"],          5, 162.0,
     540, 72, 138, 20, 70, 148, 248, 0.265, 0.818, 20, 0),
    ("Jackson Chourio", "MIL", ["CF", "LF", "OF"],          5, 165.0,
     580, 80, 155, 18, 68, 145, 248, 0.275, 0.818, 22, 0),
    ("Kyle Schwarber",  "PHI", ["LF", "1B", "OF"],          5, 168.0,
     580, 82, 140, 38, 95, 198, 278, 0.252, 0.885, 4, 0),
    ("Nolan Gorman",    "STL", ["2B", "3B"],                5, 172.0,
     520, 70, 128, 28, 80, 185, 248, 0.252, 0.838, 8, 0),
    ("Jorge Soler",     "MIA", ["LF", "DH", "OF"],          5, 175.0,
     520, 68, 128, 30, 80, 168, 258, 0.252, 0.848, 4, 0),
    ("Nico Hoerner",    "CHC", ["2B", "SS"],                5, 178.0,
     570, 75, 162, 8, 60, 82, 218, 0.292, 0.782, 18, 0),
    ("Christian Walker","HOU", ["1B"],                      5, 180.0,
     560, 72, 138, 28, 82, 165, 255, 0.258, 0.842, 4, 0),
    ("Tommy Edman",     "LAD", ["2B", "SS", "CF", "OF"],   5, 182.0,
     540, 72, 142, 12, 55, 110, 212, 0.272, 0.772, 28, 0),
    ("Max Muncy",       "LAD", ["1B", "2B", "3B"],          5, 185.0,
     530, 72, 122, 28, 75, 165, 248, 0.248, 0.848, 5, 0),
    ("Steven Kwan",     "CLE", ["LF", "CF", "OF"],          5, 188.0,
     580, 80, 172, 8, 55, 88, 218, 0.305, 0.818, 18, 0),
    ("Gleyber Torres",  "NYM", ["2B"],                      5, 190.0,
     550, 72, 148, 18, 70, 118, 242, 0.275, 0.818, 12, 0),
    ("Alejandro Kirk",  "TOR", ["C"],                       5, 193.0,
     490, 55, 132, 12, 60, 95, 218, 0.278, 0.798, 2, 0),
    ("Gabriel Moreno",  "ARI", ["C"],                       5, 195.0,
     480, 55, 128, 14, 60, 98, 218, 0.278, 0.802, 5, 0),
    ("MJ Melendez",     "KC",  ["C", "LF", "OF"],           5, 197.0,
     490, 60, 118, 18, 62, 148, 228, 0.255, 0.808, 8, 0),
    ("Logan O'Hoppe",   "LAA", ["C"],                       5, 199.0,
     480, 55, 122, 16, 60, 115, 218, 0.262, 0.802, 3, 0),
    ("Willson Contreras","STL",["C"],                       5, 202.0,
     490, 60, 125, 16, 65, 108, 228, 0.262, 0.812, 4, 0),

    # Tier 6 — ADP 200-260
    ("Josh Naylor",     "ARI", ["1B", "DH"],                6, 205.0,
     540, 65, 138, 22, 82, 128, 248, 0.268, 0.812, 3, 0),
    ("Andres Gimenez",  "CLE", ["2B"],                      6, 208.0,
     520, 68, 138, 12, 55, 108, 215, 0.272, 0.768, 16, 0),
    ("Triston Casas",   "BOS", ["1B"],                      6, 210.0,
     520, 65, 128, 22, 75, 148, 238, 0.258, 0.828, 3, 0),
    ("Spencer Torkelson","DET",["1B"],                      6, 212.0,
     530, 65, 128, 24, 78, 158, 245, 0.255, 0.828, 2, 0),
    ("Yainer Diaz",     "HOU", ["C", "1B", "DH"],           6, 214.0,
     490, 58, 128, 16, 65, 112, 222, 0.268, 0.802, 4, 0),
    ("Jonah Heim",      "TEX", ["C"],                       6, 216.0,
     460, 52, 112, 14, 55, 108, 202, 0.252, 0.778, 3, 0),
    ("Josh Smith",      "TEX", ["3B", "SS"],                6, 218.0,
     490, 62, 128, 12, 52, 112, 202, 0.272, 0.782, 10, 0),
    ("Patrick Bailey",  "SF",  ["C"],                       6, 220.0,
     440, 48, 108, 10, 48, 88, 188, 0.252, 0.752, 3, 0),
    ("Brendan Donovan", "STL", ["1B","2B","3B","LF","OF"], 6, 222.0,
     500, 62, 132, 10, 55, 98, 202, 0.272, 0.778, 8, 0),
    ("Lane Thomas",     "CLE", ["CF", "RF", "OF"],          6, 225.0,
     490, 65, 122, 16, 58, 135, 222, 0.255, 0.782, 20, 0),
    ("Bo Naylor",       "CLE", ["C", "DH"],                 6, 228.0,
     460, 55, 112, 18, 58, 148, 218, 0.252, 0.798, 4, 0),
    ("Jon Berti",       "NYM", ["2B","3B","SS","OF"],       6, 230.0,
     420, 58, 105, 6, 38, 95, 162, 0.258, 0.718, 28, 0),
    ("Brice Turang",    "MIL", ["2B", "SS"],                6, 232.0,
     490, 62, 125, 6, 42, 95, 172, 0.265, 0.728, 30, 0),
    ("Victor Scott II", "STL", ["CF", "OF"],                6, 235.0,
     440, 58, 108, 4, 32, 82, 148, 0.255, 0.698, 42, 0),
    ("Mike Trout",      "LAA", ["CF", "OF", "DH"],          6, 238.0,
     380, 52, 92, 20, 52, 118, 205, 0.272, 0.892, 4, 0),
    ("Gavin Lux",       "LAD", ["2B", "SS"],                6, 240.0,
     480, 58, 125, 10, 48, 108, 195, 0.268, 0.768, 10, 0),
    ("Ryan McMahon",    "COL", ["2B", "3B"],                6, 242.0,
     490, 58, 120, 16, 58, 115, 218, 0.252, 0.782, 10, 0),
    ("Max Kepler",      "PHI", ["RF", "LF", "OF"],          6, 244.0,
     470, 58, 120, 16, 58, 118, 215, 0.262, 0.782, 8, 0),
    ("Andrew Vaughn",   "CWS", ["1B", "LF", "DH", "OF"],  6, 246.0,
     490, 55, 128, 16, 68, 108, 218, 0.268, 0.792, 3, 0),
    ("DJ LeMahieu",     "NYY", ["1B", "2B", "3B"],          6, 248.0,
     470, 55, 125, 8, 52, 92, 190, 0.272, 0.762, 5, 0),

    # Tier 7 — ADP 250-300 (late-round targets)
    ("Masataka Yoshida","BOS", ["LF", "DH", "OF"],          7, 252.0,
     480, 55, 130, 12, 60, 85, 195, 0.282, 0.808, 4, 0),
    ("Evan Carter",     "TEX", ["CF", "LF", "OF"],          7, 255.0,
     440, 58, 112, 12, 48, 112, 185, 0.268, 0.792, 18, 0),
    ("Colt Keith",      "DET", ["2B", "3B"],                7, 258.0,
     500, 62, 130, 14, 60, 128, 212, 0.268, 0.792, 12, 0),
    ("Daulton Varsho",  "TOR", ["C", "CF", "OF"],           7, 260.0,
     440, 55, 108, 16, 55, 135, 195, 0.252, 0.778, 12, 0),
    ("Oscar Gonzalez",  "CLE", ["RF", "OF"],                7, 262.0,
     450, 52, 115, 14, 58, 95, 195, 0.262, 0.772, 4, 0),
    ("Enrique Hernandez","LAD",["2B","SS","CF","OF"],       7, 264.0,
     410, 52, 102, 14, 50, 105, 188, 0.258, 0.768, 8, 0),
    ("Gavin Sheets",    "CWS", ["1B", "LF", "DH", "OF"],  7, 266.0,
     440, 48, 110, 18, 62, 108, 205, 0.258, 0.782, 3, 0),
    ("CJ Cron",         "COL", ["1B", "DH"],                7, 268.0,
     450, 52, 112, 20, 65, 115, 215, 0.258, 0.788, 2, 0),
    ("Romy Gonzalez",   "CHW", ["SS", "2B", "3B"],          7, 270.0,
     440, 52, 108, 14, 52, 128, 195, 0.255, 0.762, 10, 0),
    ("DJ Stewart",      "FA",  ["LF", "RF", "DH", "OF"],  7, 272.0,
     410, 48, 100, 16, 55, 135, 192, 0.252, 0.768, 6, 0),
    ("Yolmer Sanchez",  "free",["2B", "3B"],                7, 274.0,
     400, 45, 98, 8, 42, 92, 158, 0.258, 0.728, 14, 0),
    ("Joey Loperfido",  "TOR", ["LF", "RF", "OF"],          7, 276.0,
     430, 55, 108, 12, 48, 118, 185, 0.262, 0.758, 20, 0),
    ("Tyrone Taylor",   "NYM", ["CF", "RF", "OF"],          7, 278.0,
     410, 50, 102, 14, 50, 118, 185, 0.258, 0.758, 12, 0),
    ("Greg Allen",      "FA",  ["CF", "LF", "OF"],          7, 280.0,
     380, 52, 95, 4, 32, 82, 135, 0.262, 0.708, 28, 0),
    ("Michael Busch",   "CHC", ["1B", "2B", "OF"],          7, 282.0,
     480, 60, 118, 18, 62, 138, 218, 0.258, 0.802, 6, 0),
    ("Nolan Jones",     "COL", ["LF", "RF", "3B", "OF"],   7, 284.0,
     420, 55, 102, 16, 55, 138, 195, 0.255, 0.808, 10, 0),
    ("Edouard Julien",  "MIN", ["2B", "DH"],                7, 286.0,
     480, 62, 118, 14, 55, 148, 198, 0.258, 0.798, 10, 0),
    ("Michael Siani",   "STL", ["CF", "OF"],                7, 288.0,
     380, 52, 92, 4, 32, 78, 128, 0.258, 0.698, 28, 0),
    ("Joc Pederson",    "SF",  ["LF", "DH", "OF"],          7, 290.0,
     440, 55, 105, 20, 60, 125, 198, 0.255, 0.812, 4, 0),
    ("David Fry",       "CLE", ["C", "1B", "3B", "OF"],   7, 292.0,
     410, 48, 100, 14, 52, 118, 185, 0.255, 0.778, 5, 0),
    ("Tyler O'Neill",   "BOS", ["LF", "RF", "CF", "OF"],  7, 294.0,
     420, 52, 100, 18, 55, 148, 198, 0.252, 0.778, 10, 0),
    ("Henry Davis",     "PIT", ["C"],                       7, 296.0,
     420, 48, 100, 14, 52, 128, 185, 0.252, 0.762, 4, 0),
    ("Avisail Garcia",  "free",["LF", "RF", "DH", "OF"],  7, 298.0,
     380, 42, 95, 12, 48, 105, 172, 0.258, 0.762, 5, 0),
    ("Nick Ahmed",      "ARI", ["SS"],                      8, 300.0,
     360, 40, 88, 8, 38, 92, 145, 0.252, 0.718, 8, 0),
]

_PITCHER_RAW = [
    # Tier 1 — Ace SPs / Elite Closers (pick 14-30)
    # (name, team, positions, tier, adp, ip, w, l, sv, bs, qs, k, era, whip, k9, hr, nsv)
    ("Spencer Strider",  "ATL", ["SP"],     1, 13.0,
     175, 14, 5, 0, 0, 21, 242, 2.68, 0.94, 12.5, 16, 0),
    ("Gerrit Cole",      "NYY", ["SP"],     1, 14.0,
     188, 14, 6, 0, 0, 23, 226, 2.80, 1.00, 10.8, 18, 0),
    ("Paul Skenes",      "PIT", ["SP"],     1, 19.0,
     172, 12, 7, 0, 0, 20, 222, 2.88, 1.02, 11.6, 16, 0),
    ("Zack Wheeler",     "PHI", ["SP"],     1, 20.0,
     195, 14, 6, 0, 0, 24, 216, 2.98, 1.04, 9.9, 18, 0),
    ("Emmanuel Clase",   "CLE", ["RP"],     1, 23.0,
     70, 4, 2, 38, 4, 0, 84, 2.18, 0.88, 10.8, 4, 34),
    ("Logan Webb",       "SF",  ["SP"],     1, 25.0,
     200, 14, 7, 0, 0, 24, 196, 3.08, 1.07, 8.8, 16, 0),
    ("Corbin Burnes",    "BAL", ["SP"],     1, 26.0,
     192, 13, 7, 0, 0, 22, 208, 3.08, 1.05, 9.7, 17, 0),
    ("Josh Hader",       "HOU", ["RP"],     1, 29.0,
     65, 3, 3, 35, 6, 0, 90, 2.38, 0.92, 12.5, 4, 29),
    ("Dylan Cease",      "SD",  ["SP"],     1, 33.0,
     182, 12, 8, 0, 0, 20, 202, 3.38, 1.14, 10.0, 19, 0),
    ("Kevin Gausman",    "TOR", ["SP"],     1, 34.0,
     188, 12, 8, 0, 0, 22, 196, 3.18, 1.04, 9.4, 19, 0),

    # Tier 2 — SP2/Strong Closer (pick 31-72)
    ("Edwin Diaz",       "NYM", ["RP"],     2, 37.0,
     65, 3, 3, 32, 5, 0, 92, 2.48, 0.94, 12.7, 4, 27),
    ("Framber Valdez",   "HOU", ["SP"],     2, 38.0,
     195, 13, 8, 0, 0, 23, 178, 3.22, 1.18, 8.2, 15, 0),
    ("Tarik Skubal",     "DET", ["SP"],     2, 39.0,
     180, 13, 6, 0, 0, 22, 208, 3.02, 1.06, 10.4, 17, 0),
    ("Félix Bautista",   "BAL", ["RP"],     2, 43.0,
     62, 3, 3, 30, 5, 0, 82, 2.58, 0.98, 11.9, 5, 25),
    ("Shane Bieber",     "CLE", ["SP"],     2, 46.0,
     178, 12, 7, 0, 0, 21, 192, 3.12, 1.04, 9.7, 17, 0),
    ("Sonny Gray",       "STL", ["SP"],     2, 47.0,
     175, 12, 7, 0, 0, 20, 188, 3.22, 1.10, 9.7, 18, 0),
    ("Camilo Doval",     "SF",  ["RP"],     2, 49.0,
     65, 3, 4, 30, 7, 0, 74, 2.78, 1.04, 10.2, 5, 23),
    ("Aaron Nola",       "PHI", ["SP"],     2, 51.0,
     192, 13, 8, 0, 0, 22, 198, 3.48, 1.12, 9.3, 22, 0),
    ("Tyler Glasnow",    "LAD", ["SP"],     2, 53.0,
     162, 11, 6, 0, 0, 19, 198, 3.18, 1.06, 11.0, 17, 0),
    ("Yordan Guzman",    "TB",  ["SP"],     2, 57.0,
     172, 11, 7, 0, 0, 19, 185, 3.38, 1.14, 9.7, 18, 0),
    ("Blake Snell",      "SF",  ["SP"],     2, 59.0,
     158, 10, 7, 0, 0, 17, 195, 3.22, 1.12, 11.1, 16, 0),
    ("Jordan Romano",    "TOR", ["RP"],     2, 61.0,
     60, 3, 3, 28, 6, 0, 68, 2.68, 1.02, 10.2, 5, 22),
    ("Andrés Muñoz",     "SEA", ["RP"],     2, 63.0,
     62, 3, 3, 28, 6, 0, 78, 2.48, 0.98, 11.3, 4, 22),
    ("Ryan Helsley",     "STL", ["RP"],     2, 66.0,
     62, 3, 4, 28, 6, 0, 78, 2.78, 1.04, 11.3, 5, 22),

    # Tier 3 — SP3/Middle Closer (pick 73-120)
    ("Pablo Lopez",      "MIN", ["SP"],     3, 73.0,
     185, 11, 8, 0, 0, 20, 185, 3.58, 1.15, 9.0, 20, 0),
    ("Max Fried",        "NYY", ["SP"],     3, 76.0,
     180, 11, 8, 0, 0, 20, 182, 3.48, 1.14, 9.1, 18, 0),
    ("Hunter Greene",    "CIN", ["SP"],     3, 78.0,
     172, 10, 8, 0, 0, 18, 195, 3.58, 1.15, 10.2, 20, 0),
    ("George Kirby",     "SEA", ["SP"],     3, 80.0,
     185, 11, 7, 0, 0, 20, 178, 3.42, 1.09, 8.7, 18, 0),
    ("José Berríos",     "TOR", ["SP"],     3, 82.0,
     188, 12, 9, 0, 0, 20, 188, 3.62, 1.18, 9.0, 21, 0),
    ("Grayson Rodriguez","BAL", ["SP"],     3, 84.0,
     175, 11, 8, 0, 0, 19, 192, 3.48, 1.14, 9.9, 19, 0),
    ("Kodai Senga",      "NYM", ["SP"],     3, 86.0,
     165, 10, 7, 0, 0, 18, 188, 3.28, 1.08, 10.3, 16, 0),
    ("Tanner Bibee",     "CLE", ["SP"],     3, 88.0,
     175, 11, 7, 0, 0, 19, 182, 3.48, 1.16, 9.4, 18, 0),
    ("Devin Williams",   "NYY", ["RP"],     3, 90.0,
     60, 4, 3, 25, 7, 0, 78, 2.88, 1.06, 11.7, 5, 18),
    ("Clay Holmes",      "CLE", ["RP"],     3, 93.0,
     62, 4, 4, 24, 7, 0, 72, 2.98, 1.12, 10.4, 5, 17),
    ("Jordan Hicks",     "TOR", ["RP"],     3, 95.0,
     62, 4, 3, 26, 6, 0, 68, 2.88, 1.08, 9.9, 5, 20),
    ("Reynaldo Lopez",   "ATL", ["RP"],     3, 97.0,
     62, 4, 3, 26, 6, 0, 72, 2.88, 1.05, 10.4, 4, 20),
    ("Michael King",     "SD",  ["SP", "RP"],3, 100.0,
     155, 9, 7, 0, 0, 15, 172, 3.68, 1.14, 10.0, 18, 0),
    ("Cristopher Sanchez","PHI",["SP"],     3, 102.0,
     175, 11, 8, 0, 0, 18, 165, 3.62, 1.18, 8.5, 18, 0),
    ("Brayan Bello",     "BOS", ["SP"],     3, 105.0,
     178, 10, 9, 0, 0, 18, 175, 3.78, 1.22, 8.8, 19, 0),

    # Tier 4 — SP4/Streaming/Closer Handcuff (pick 121-168)
    ("Shota Imanaga",    "CHC", ["SP"],     4, 112.0,
     175, 10, 8, 0, 0, 18, 178, 3.62, 1.14, 9.2, 17, 0),
    ("Joe Ryan",         "MIN", ["SP"],     4, 115.0,
     178, 10, 8, 0, 0, 18, 182, 3.72, 1.14, 9.2, 20, 0),
    ("Eury Pérez",       "MIA", ["SP"],     4, 118.0,
     162, 9, 8, 0, 0, 17, 185, 3.68, 1.18, 10.3, 17, 0),
    ("MacKenzie Gore",   "WSH", ["SP"],     4, 120.0,
     168, 9, 9, 0, 0, 17, 182, 3.78, 1.20, 9.8, 18, 0),
    ("Nick Lodolo",      "CIN", ["SP"],     4, 123.0,
     162, 9, 8, 0, 0, 16, 175, 3.78, 1.18, 9.7, 17, 0),
    ("Gavin Stone",      "LAD", ["SP"],     4, 125.0,
     165, 9, 7, 0, 0, 16, 165, 3.88, 1.18, 9.0, 17, 0),
    ("Clarke Schmidt",   "NYY", ["SP"],     4, 128.0,
     168, 10, 8, 0, 0, 17, 172, 3.82, 1.18, 9.2, 18, 0),
    ("Logan Gilbert",    "SEA", ["SP"],     4, 130.0,
     182, 10, 9, 0, 0, 18, 178, 3.88, 1.18, 8.8, 19, 0),
    ("DL Hall",          "MIL", ["SP", "RP"],4, 133.0,
     155, 9, 8, 0, 0, 15, 172, 3.78, 1.18, 10.0, 17, 0),
    ("Kyle Freeland",    "COL", ["SP"],     4, 136.0,
     175, 9, 10, 0, 0, 16, 152, 4.18, 1.28, 7.8, 22, 0),
    ("Trevor Rogers",    "MIA", ["SP"],     4, 138.0,
     162, 9, 9, 0, 0, 15, 168, 3.98, 1.25, 9.3, 18, 0),
    ("Justin Verlander", "HOU", ["SP"],     4, 140.0,
     155, 10, 7, 0, 0, 16, 155, 3.68, 1.15, 9.0, 18, 0),
    ("Chris Sale",       "ATL", ["SP"],     4, 143.0,
     168, 10, 7, 0, 0, 17, 188, 3.58, 1.10, 10.1, 17, 0),
    ("Ryan Pressly",     "HOU", ["RP"],     4, 146.0,
     58, 3, 3, 20, 6, 0, 62, 3.18, 1.12, 9.6, 6, 14),
    ("Robert Suarez",    "SD",  ["RP"],     4, 148.0,
     58, 3, 4, 22, 6, 0, 64, 2.98, 1.06, 9.9, 5, 16),
    ("Raisel Iglesias",  "LAA", ["RP"],     4, 150.0,
     58, 3, 4, 22, 6, 0, 68, 2.88, 1.08, 10.6, 5, 16),

    # Tier 5 — Streamers / Speculative (pick 169-230)
    ("Patrick Corbin",   "WSH", ["SP"],     5, 165.0,
     158, 8, 12, 0, 0, 13, 138, 4.58, 1.35, 7.9, 22, 0),
    ("Lance Lynn",       "STL", ["SP"],     5, 168.0,
     165, 9, 10, 0, 0, 15, 155, 4.18, 1.28, 8.5, 22, 0),
    ("Freddy Peralta",   "MIL", ["SP"],     5, 170.0,
     155, 8, 8, 0, 0, 14, 175, 3.88, 1.18, 10.2, 17, 0),
    ("Josh Walker",      "NYM", ["SP", "RP"],5, 175.0,
     145, 8, 8, 0, 0, 13, 158, 3.98, 1.24, 9.8, 17, 0),
    ("Matt Brash",       "SEA", ["RP"],     5, 178.0,
     55, 3, 3, 18, 6, 0, 62, 3.08, 1.12, 10.1, 6, 12),
    ("Kenley Jansen",    "BOS", ["RP"],     5, 180.0,
     55, 3, 4, 18, 7, 0, 62, 3.18, 1.15, 10.1, 6, 11),
    ("Pete Fairbanks",   "TB",  ["RP"],     5, 182.0,
     55, 3, 3, 20, 6, 0, 65, 2.98, 1.08, 10.6, 5, 14),
    ("Alex Vesia",       "LAD", ["RP"],     5, 185.0,
     55, 3, 3, 12, 5, 0, 68, 2.98, 1.10, 11.1, 5, 7),

    # Tier 5 cont — SP streamers / handcuff closers (ADP 185-240)
    ("Mason Miller",     "OAK", ["RP"],     5, 188.0,
     60, 3, 3, 28, 5, 0, 80, 2.58, 0.95, 12.0, 4, 23),
    ("Luis Castillo",    "SEA", ["SP"],     5, 190.0,
     185, 11, 8, 0, 0, 19, 185, 3.52, 1.15, 9.0, 19, 0),
    ("Brady Singer",     "KC",  ["SP"],     5, 193.0,
     182, 10, 9, 0, 0, 18, 175, 3.68, 1.18, 8.7, 20, 0),
    ("Mitch Keller",     "PIT", ["SP"],     5, 196.0,
     185, 10, 9, 0, 0, 18, 178, 3.72, 1.19, 8.7, 20, 0),
    ("Hunter Brown",     "HOU", ["SP"],     5, 198.0,
     172, 10, 8, 0, 0, 17, 185, 3.58, 1.16, 9.7, 18, 0),
    ("David Bednar",     "PIT", ["RP"],     5, 200.0,
     58, 3, 4, 24, 7, 0, 68, 2.98, 1.06, 10.6, 5, 17),
    ("Kyle Finnegan",    "WSH", ["RP"],     5, 203.0,
     58, 3, 4, 26, 7, 0, 62, 3.08, 1.12, 9.6, 6, 19),
    ("Paul Sewald",      "ARI", ["RP"],     5, 205.0,
     55, 3, 3, 24, 6, 0, 68, 2.88, 1.04, 11.1, 4, 18),
    ("Hunter Harvey",    "WSH", ["RP"],     5, 208.0,
     55, 3, 4, 22, 6, 0, 62, 3.08, 1.08, 10.1, 5, 16),
    ("Nestor Cortes",    "MIL", ["SP"],     5, 210.0,
     168, 9, 8, 0, 0, 16, 165, 3.78, 1.18, 8.8, 17, 0),
    ("Zach Eflin",       "TB",  ["SP"],     5, 212.0,
     178, 10, 9, 0, 0, 18, 168, 3.82, 1.19, 8.5, 20, 0),
    ("Bailey Ober",      "MIN", ["SP"],     5, 215.0,
     175, 10, 9, 0, 0, 17, 172, 3.78, 1.17, 8.8, 19, 0),
    ("Jose Alvarado",    "PHI", ["RP"],     5, 218.0,
     55, 3, 3, 20, 6, 0, 68, 2.98, 1.10, 11.1, 5, 14),
    ("Ryan Walker",      "SF",  ["RP"],     5, 220.0,
     55, 3, 3, 18, 6, 0, 62, 3.18, 1.14, 10.1, 5, 12),
    ("A.J. Minter",      "ATL", ["RP"],     5, 222.0,
     58, 3, 3, 8, 4, 0, 68, 2.98, 1.08, 10.6, 5, 4),
    ("Ranger Suarez",    "PHI", ["SP"],     5, 225.0,
     172, 10, 8, 0, 0, 17, 162, 3.68, 1.20, 8.5, 16, 0),
    ("Nathan Eovaldi",   "TEX", ["SP"],     5, 228.0,
     158, 9, 8, 0, 0, 16, 155, 3.82, 1.18, 8.8, 18, 0),
    ("Kyle Harrison",    "SF",  ["SP"],     5, 230.0,
     162, 9, 8, 0, 0, 15, 172, 3.88, 1.20, 9.6, 17, 0),

    # Tier 6 — Mid streamers / closer handcuffs (ADP 230-270)
    ("Max Meyer",        "MIA", ["SP"],     6, 232.0,
     155, 8, 8, 0, 0, 14, 168, 3.98, 1.24, 9.8, 17, 0),
    ("Braxton Garrett",  "MIA", ["SP"],     6, 234.0,
     158, 9, 9, 0, 0, 15, 155, 3.98, 1.24, 8.8, 18, 0),
    ("Jose Leclerc",     "TEX", ["RP"],     6, 236.0,
     52, 3, 3, 16, 6, 0, 58, 3.28, 1.15, 10.0, 6, 10),
    ("Kevin Ginkel",     "ARI", ["RP"],     6, 238.0,
     52, 3, 3, 16, 6, 0, 58, 3.18, 1.12, 10.0, 5, 10),
    ("Evan Phillips",    "LAD", ["RP"],     6, 240.0,
     55, 3, 3, 14, 5, 0, 65, 2.98, 1.06, 10.6, 4, 9),
    ("Carlos Rodon",     "NYY", ["SP"],     6, 242.0,
     155, 9, 8, 0, 0, 15, 178, 3.78, 1.20, 10.3, 17, 0),
    ("Triston McKenzie", "CLE", ["SP"],     6, 244.0,
     158, 9, 9, 0, 0, 14, 168, 3.98, 1.22, 9.6, 18, 0),
    ("JP Sears",         "OAK", ["SP"],     6, 246.0,
     165, 9, 10, 0, 0, 15, 152, 4.08, 1.24, 8.3, 20, 0),
    ("Aaron Civale",     "TB",  ["SP"],     6, 248.0,
     162, 9, 9, 0, 0, 15, 148, 4.02, 1.22, 8.2, 19, 0),
    ("Seranthony Dominguez","PHI",["RP"],   6, 250.0,
     52, 3, 3, 14, 5, 0, 58, 3.18, 1.14, 10.0, 5, 9),
    ("Matt Strahm",      "PHI", ["SP","RP"],6, 252.0,
     145, 8, 8, 0, 0, 12, 155, 3.98, 1.20, 9.6, 16, 0),
    ("Aroldis Chapman",  "PIT", ["RP"],     6, 254.0,
     50, 3, 4, 12, 5, 0, 65, 3.28, 1.18, 11.7, 6, 7),
    ("Jake Irvin",       "WSH", ["SP"],     6, 256.0,
     162, 9, 10, 0, 0, 15, 148, 4.18, 1.25, 8.2, 20, 0),
    ("Jameson Taillon",  "CHC", ["SP"],     6, 258.0,
     158, 9, 9, 0, 0, 14, 142, 4.08, 1.22, 8.1, 19, 0),
    ("DL Hall",          "MIL", ["SP","RP"],6, 260.0,
     148, 8, 7, 0, 0, 13, 162, 3.88, 1.20, 9.9, 16, 0),
    # Josh Hader — NOT duplicated here; already in Tier 1 at ADP 29

    # Tier 7 — Deep streamers / handcuffs / speculative (ADP 260-310)
    ("Phil Maton",       "HOU", ["RP"],     7, 264.0,
     50, 3, 3, 10, 5, 0, 55, 3.28, 1.16, 9.9, 5, 5),
    ("Craig Kimbrel",    "PHI", ["RP"],     7, 266.0,
     48, 3, 4, 14, 6, 0, 58, 3.48, 1.18, 10.9, 6, 8),
    ("Andrew Heaney",    "TEX", ["SP"],     7, 268.0,
     145, 8, 8, 0, 0, 12, 162, 3.98, 1.20, 10.1, 16, 0),
    ("Spencer Turnbull", "PHI", ["SP"],     7, 270.0,
     148, 8, 8, 0, 0, 12, 155, 3.92, 1.19, 9.5, 16, 0),
    ("Ben Lively",       "CLE", ["SP"],     7, 272.0,
     162, 9, 9, 0, 0, 14, 145, 4.12, 1.24, 8.1, 20, 0),
    ("Giovanny Gallegos","STL", ["RP"],     7, 274.0,
     50, 2, 3, 10, 5, 0, 58, 3.38, 1.16, 10.4, 5, 5),
    ("Kyle Muller",      "OAK", ["SP"],     7, 276.0,
     148, 8, 9, 0, 0, 13, 152, 4.18, 1.26, 9.2, 18, 0),
    ("Mitchell Parker",  "WSH", ["SP"],     7, 278.0,
     148, 8, 9, 0, 0, 12, 148, 4.08, 1.26, 9.0, 18, 0),
    ("Reid Detmers",     "LAA", ["SP"],     7, 280.0,
     148, 7, 9, 0, 0, 12, 152, 4.02, 1.22, 9.2, 17, 0),
    ("Colin Rea",        "MIL", ["SP"],     7, 282.0,
     155, 8, 9, 0, 0, 13, 138, 4.22, 1.26, 8.0, 20, 0),
    ("Taj Bradley",      "TB",  ["SP"],     7, 284.0,
     148, 8, 9, 0, 0, 13, 155, 4.08, 1.22, 9.4, 18, 0),
    ("Joe Boyle",        "OAK", ["SP"],     7, 286.0,
     140, 7, 9, 0, 0, 11, 145, 4.18, 1.28, 9.3, 17, 0),
    ("Louie Varland",    "MIN", ["SP"],     7, 288.0,
     145, 8, 9, 0, 0, 12, 148, 4.18, 1.28, 9.2, 18, 0),
    ("Gavin Williams",   "CLE", ["SP"],     7, 290.0,
     145, 8, 9, 0, 0, 12, 152, 3.98, 1.22, 9.4, 17, 0),
    ("Reese Olson",      "DET", ["SP"],     7, 292.0,
     148, 8, 9, 0, 0, 12, 148, 4.08, 1.24, 9.0, 18, 0),
    ("Tommy Henry",      "ARI", ["SP"],     7, 294.0,
     140, 7, 9, 0, 0, 11, 138, 4.22, 1.28, 8.9, 18, 0),
    ("Ryan Pepiot",      "TB",  ["SP"],     7, 296.0,
     145, 8, 8, 0, 0, 13, 148, 3.98, 1.22, 9.2, 17, 0),
    ("Yerry De Los Santos","OAK",["RP"],    7, 298.0,
     48, 2, 3, 10, 5, 0, 55, 3.38, 1.18, 10.3, 5, 5),
    ("Cade Cavalli",     "WSH", ["SP"],     7, 300.0,
     138, 7, 9, 0, 0, 11, 145, 4.18, 1.26, 9.5, 17, 0),
    ("Tyler Wells",      "BAL", ["SP","RP"],7, 302.0,
     145, 8, 9, 0, 0, 12, 148, 4.08, 1.24, 9.2, 18, 0),
    # Grayson Rodriguez — NOT duplicated here; already in Tier 3 at ADP 84
]


# ---------------------------------------------------------------------------
# Build the board — compute z-scores and rank
# ---------------------------------------------------------------------------

def _parse_batter(row: tuple) -> dict:
    (name, team, positions, tier, adp,
     pa, r, h, hr, rbi, k_bat, tb, avg, ops, nsb, _) = row
    slg = ops - 0.330
    return {
        "id": name.lower().replace(" ", "_").replace(".", "").replace("'", ""),
        "name": name, "team": team, "positions": positions,
        "type": "batter", "tier": tier, "adp": adp,
        "proj": {
            "pa": pa, "r": r, "h": h, "hr": hr, "rbi": rbi,
            "k_bat": k_bat, "tb": tb, "avg": avg, "ops": ops,
            "nsb": nsb, "slg": slg,
        },
        "z_score": 0.0, "rank": 0, "cat_scores": {},
    }


def _parse_pitcher(row: tuple) -> dict:
    (name, team, positions, tier, adp,
     ip, w, l, sv, bs, qs, k, era, whip, k9, hr_pit, nsv) = row
    return {
        "id": name.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("é", "e").replace("á", "a").replace("ó", "o").replace("ú", "u").replace("í", "i"),
        "name": name, "team": team, "positions": positions,
        "type": "pitcher", "tier": tier, "adp": adp,
        "proj": {
            "ip": ip, "w": w, "l": l, "sv": sv, "bs": bs,
            "qs": qs, "k_pit": k, "era": era, "whip": whip,
            "k9": k9, "hr_pit": hr_pit, "nsv": nsv,
        },
        "z_score": 0.0, "rank": 0, "cat_scores": {},
    }


def _zscore(value: float, values: list[float], direction: int = 1) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std < 1e-9:
        return 0.0
    return ((value - mean) / std) * direction


def _compute_zscores(batters: list[dict], pitchers: list[dict]) -> None:
    """Compute and assign z-scores for all players in-place."""

    # Batter pools
    bat_pools = {
        "r":     ([p["proj"]["r"]     for p in batters], 1),
        "h":     ([p["proj"]["h"]     for p in batters], 1),
        "hr":    ([p["proj"]["hr"]    for p in batters], 1),
        "rbi":   ([p["proj"]["rbi"]   for p in batters], 1),
        "k_bat": ([p["proj"]["k_bat"] for p in batters], -1),   # negative
        "tb":    ([p["proj"]["tb"]    for p in batters], 1),
        "avg":   ([p["proj"]["avg"]   for p in batters], 1),
        "ops":   ([p["proj"]["ops"]   for p in batters], 1),
        "nsb":   ([p["proj"]["nsb"]   for p in batters], 1),
    }
    bat_weights = {
        "r": 1.0, "h": 0.9, "hr": 1.1, "rbi": 1.1, "k_bat": 0.8,
        "tb": 1.0, "avg": 1.1, "ops": 1.2, "nsb": 1.4,
    }
    for p in batters:
        total = 0.0
        cat_scores = {}
        for cat, (pool, direction) in bat_pools.items():
            z = _zscore(p["proj"][cat], pool, direction)
            w = bat_weights[cat]
            wz = z * w
            cat_scores[cat] = round(wz, 3)
            total += wz
        p["z_score"] = round(total, 3)
        p["cat_scores"] = cat_scores

    # Pitcher pools
    pit_pools = {
        "w":      ([p["proj"]["w"]      for p in pitchers], 1),
        "l":      ([p["proj"]["l"]      for p in pitchers], -1),   # negative
        "hr_pit": ([p["proj"]["hr_pit"] for p in pitchers], -1),   # negative
        "k_pit":  ([p["proj"]["k_pit"]  for p in pitchers], 1),
        "era":    ([p["proj"]["era"]    for p in pitchers], -1),   # negative
        "whip":   ([p["proj"]["whip"]   for p in pitchers], -1),   # negative
        "k9":     ([p["proj"]["k9"]     for p in pitchers], 1),
        "qs":     ([p["proj"]["qs"]     for p in pitchers], 1),
        "nsv":    ([p["proj"]["nsv"]    for p in pitchers], 1),
    }
    pit_weights = {
        "w": 1.1, "l": 0.8, "hr_pit": 0.7, "k_pit": 1.1,
        "era": 1.1, "whip": 1.1, "k9": 1.0, "qs": 1.0, "nsv": 1.5,
    }
    for p in pitchers:
        total = 0.0
        cat_scores = {}
        for cat, (pool, direction) in pit_pools.items():
            z = _zscore(p["proj"][cat], pool, direction)
            w = pit_weights[cat]
            wz = z * w
            cat_scores[cat] = round(wz, 3)
            total += wz
        p["z_score"] = round(total, 3)
        p["cat_scores"] = cat_scores


def build_board() -> list[dict]:
    """
    Build and return the full ranked player board.
    Players are sorted by z_score (within position type), then merged by ADP.
    """
    batters = [_parse_batter(r) for r in _BATTER_RAW]
    pitchers = [_parse_pitcher(r) for r in _PITCHER_RAW]

    _compute_zscores(batters, pitchers)

    # Rank batters and pitchers separately by z_score
    batters.sort(key=lambda p: p["z_score"], reverse=True)
    pitchers.sort(key=lambda p: p["z_score"], reverse=True)
    for i, p in enumerate(batters, 1):
        p["bat_rank"] = i
    for i, p in enumerate(pitchers, 1):
        p["pit_rank"] = i

    # Merge and sort by ADP for overall rank
    all_players = batters + pitchers
    all_players.sort(key=lambda p: p["adp"])

    # Deduplicate by player ID — keep the first occurrence (lowest ADP)
    seen_ids: set[str] = set()
    deduped = []
    for p in all_players:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            deduped.append(p)

    for i, p in enumerate(deduped, 1):
        p["rank"] = i

    return deduped


# Singleton — built once, reused per process
_BOARD: Optional[list[dict]] = None


def invalidate_board() -> None:
    """Force board rebuild on next call (use after loading real CSVs)."""
    global _BOARD
    _BOARD = None


def get_board(apply_park_factors: bool = True) -> list[dict]:
    """
    Return the full ranked player board.

    Priority:
    1. Real Steamer/ZiPS CSV data (if data/projections/ CSVs are present)
    2. Hardcoded estimates (fallback — always available)

    Park factors and risk adjustments are applied on top of either source.
    """
    global _BOARD
    if _BOARD is None:
        # Try real projection data first
        try:
            from backend.fantasy_baseball.projections_loader import load_full_board
            real_board = load_full_board()
            if real_board:
                _BOARD = real_board
        except Exception:
            pass

        if _BOARD is None:
            _BOARD = build_board()

        # Apply park factors and risk flags to whichever board we have
        if apply_park_factors:
            try:
                from backend.fantasy_baseball.ballpark_factors import annotate_board
                annotate_board(_BOARD)
            except Exception:
                pass

        # Stamp keeper flags
        annotate_keepers(_BOARD)

    return _BOARD


def get_player_by_name(name: str) -> Optional[dict]:
    board = get_board()
    name_lower = name.lower()
    for p in board:
        if name_lower in p["name"].lower():
            return p
    return None


# ---------------------------------------------------------------------------
# Keeper configuration — my keepers for this season
# ---------------------------------------------------------------------------

MY_KEEPERS: dict[str, int] = {
    "juan_soto": 1,   # Keep Juan Soto, costs Round 1
}

# All 14 league-wide keepers (all teams). Used to pre-filter the value board
# before the Yahoo roster API sweep fires at 19:00. Verified from Yahoo lock
# screen 2026-03-23.
ALL_LEAGUE_KEEPERS: frozenset[str] = frozenset({
    "aaron_judge",        # ChippaJone
    "shohei_ohtani",      # Marte Partay
    "bobby_witt_jr",      # Bartolo's Colon
    "juan_soto",          # Lindor Truffles (us)
    "elly_de_la_cruz",    # Mendoza Line
    "kyle_tucker",        # Juiced Balls
    "jose_ramirez",       # Game Blausers
    "ronald_acuna_jr",    # Juiced Balls
    "julio_rodriguez",    # Damn the Torpedoes
    "corbin_carroll",     # Slap Dick Prospects
    "fernando_tatis_jr",  # Mendoza Line
    "francisco_lindor",   # ChippaJone
    "nick_kurtz",         # High&TightyWhitey's
    "jackson_merrill",    # High&TightyWhitey's
})


def annotate_keepers(board: list[dict]) -> None:
    """Stamp is_keeper / keeper_round onto keeper players (in-place)."""
    for p in board:
        if p["id"] in MY_KEEPERS:
            p["is_keeper"] = True
            p["keeper_round"] = MY_KEEPERS[p["id"]]
        elif p["id"] in ALL_LEAGUE_KEEPERS:
            p["is_keeper"] = True
            p["keeper_round"] = None  # other team's keeper — round unknown
        else:
            p.setdefault("is_keeper", False)
            p.setdefault("keeper_round", None)


def available_players(drafted_ids: set[str]) -> list[dict]:
    """Return players not yet drafted and not a league keeper."""
    excluded = ALL_LEAGUE_KEEPERS | drafted_ids
    return [p for p in get_board() if p["id"] not in excluded]


# ---------------------------------------------------------------------------
# Position baseline z-scores (conservative — median of bottom half of board)
# Used as proxy for call-ups / undrafted players not in PLAYER_BOARD.
# Derived from get_board() distribution; update each spring.
# ---------------------------------------------------------------------------

_POSITION_BASELINE_Z: dict[str, float] = {
    "SP": -0.5,
    "RP": -0.3,
    "C":  -1.5,   # Catchers are scarce — even fringe C has roster value
    "1B": -0.8,
    "2B": -0.8,
    "3B": -0.8,
    "SS": -0.8,
    "OF": -0.8,
    "LF": -0.8,
    "CF": -0.8,
    "RF": -0.8,
    "DH": -1.0,
    "P":  -0.5,   # Generic pitcher
}
_DEFAULT_BASELINE_Z = -1.0  # Unknown position

# Runtime cache for on-the-fly projections (cleared on process restart)
_projection_cache: dict[str, dict] = {}


def get_or_create_projection(yahoo_player: dict) -> dict:
    """
    Return a board-compatible dict for any Yahoo player, whether they are
    on the draft board or not.

    For board players: returns the existing entry (rich projections).
    For unknown players (call-ups, recent adds): creates a minimal entry
    using position-average z-score as proxy.  The proxy is intentionally
    conservative — these players are unproven at MLB level.

    Args:
        yahoo_player: dict from YahooFantasyClient (has name, player_key,
                      positions, team, percent_owned, etc.)

    Returns:
        board-compatible dict with at minimum: name, z_score, positions,
        cat_scores (empty for proxy), type.
    """
    name = (yahoo_player.get("name") or "").strip()
    player_key = yahoo_player.get("player_key") or ""

    # 1. Check runtime cache first (avoids repeated lookups)
    if player_key and player_key in _projection_cache:
        return _projection_cache[player_key]

    # 2. Check board by exact name match
    board = get_board()
    board_by_name = {p["name"].lower(): p for p in board}
    entry = board_by_name.get(name.lower())

    if entry:
        if player_key:
            _projection_cache[player_key] = entry
        return entry

    # 3. Fuzzy name match — handles "José" vs "Jose", suffixes, etc.
    import difflib as _difflib
    name_lower = name.lower()
    clean_name = "".join(c for c in name_lower if c.isalnum() or c == " ")
    for board_name, board_entry in board_by_name.items():
        # Strip accents / punctuation for comparison
        clean_board = "".join(c for c in board_name if c.isalnum() or c == " ")
        if clean_board == clean_name:
            if player_key:
                _projection_cache[player_key] = board_entry
            return board_entry

    # 3b. Similarity match — handles "Christopher" vs "Cristopher", etc.
    best_ratio = 0.0
    best_entry = None
    for board_name, board_entry in board_by_name.items():
        clean_board = "".join(c for c in board_name if c.isalnum() or c == " ")
        ratio = _difflib.SequenceMatcher(None, clean_name, clean_board).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_entry = board_entry
    if best_ratio >= 0.90 and best_entry is not None:
        if player_key:
            _projection_cache[player_key] = best_entry
        return best_entry

    # 4. Not on board — build proxy entry
    positions = yahoo_player.get("positions") or []
    primary_pos = positions[0] if positions else ""

    # Infer type from position
    player_type = "pitcher" if primary_pos in ("SP", "RP", "P") else "batter"

    # Use position baseline z_score
    proxy_z = _POSITION_BASELINE_Z.get(primary_pos, _DEFAULT_BASELINE_Z)

    proxy = {
        "id": player_key or name.lower().replace(" ", "_"),
        "name": name,
        "team": yahoo_player.get("team") or "",
        "positions": positions,
        "type": player_type,
        "tier": 10,
        "rank": 9999,
        "adp": 9999.0,
        "z_score": proxy_z,
        "cat_scores": {},  # No per-category data for unknown players
        "proj": {},
        "is_keeper": False,
        "keeper_round": None,
        "is_proxy": True,  # Flag so callers know this is estimated
    }

    if player_key:
        _projection_cache[player_key] = proxy
    return proxy


# ---------------------------------------------------------------------------
# CLI — print top 50
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    board = get_board()
    print(f"\n{'Rank':>4} {'Tier':>4} {'Name':<24} {'Team':>4} {'Pos':<18} {'Type':>7} {'ADP':>6} {'Z-Score':>8}")
    print("-" * 85)
    for p in board[:60]:
        pos_str = "/".join(p["positions"][:3])
        print(
            f"{p['rank']:>4} {p['tier']:>4}  {p['name']:<24} {p['team']:>4} "
            f"{pos_str:<18} {p['type']:>7} {p['adp']:>6.1f} {p['z_score']:>+8.2f}"
        )
