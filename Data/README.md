# Data Description

## Summary of data

Here, you'll find a summary of each data set in the 2024 Data Bowl, a list of key variables to join on, and a description of each variable. The tracking data is provided by the NFL Next Gen Stats team. The pff_missedTackles column in the tackles data is provided by Pro Football Focus.
Supplemental Data

The 2024 Big Data Bowl allows participants to use supplemental NFL data as long as it is free and publicly available to all participants. Examples of sources that could be used include nflverse and Pro Football Reference. Please note that the gameId and playId of the Big Data Bowl data merges with the old_game_id and play_id of nflverse's play-by-play data.

## File descriptions

* Game data: The games.csv contains the teams playing in each game. The key variable is gameId.
* Play data: The plays.csv file contains play-level information from each game. The key variables are gameId and playId.
* Player data: The players.csv file contains player-level information from players that participated in any of the tracking data files. The key variable is nflId.
* Tackles data: The tackles.csv file contains player-level tackle information for each game and play. The key variables are gameId, playId, and nflId.
* Tracking data: Files tracking_week_[week].csv contain player tracking data from week number [week]. The key variables are gameId, playId, and nflId.

### Game data

* gameId: Game identifier, unique (numeric)
* season: Season of game
* week: Week of game
* gameDate: Game Date (time, mm/dd/yyyy)
* gameTimeEastern: Start time of game (time, HH:MM:SS, EST)
* homeTeamAbbr: Home team three-letter code (text)
* visitorTeamAbbr: Visiting team three-letter code (text)
* homeFinalScore: The total amount of points scored by the home team in the game (numeric)
* visitorFinalScore: The total amount of points scored by the visiting team in the game (numeric)

### Play data

* gameId: Game identifier, unique (numeric)
* playId: Play identifier, not unique across games (numeric)
* ballCarrierId: The nflId of the ball carrier (receiver of the handoff, receiver of pass or the QB scrambling) on the play. This is the player that the defense is attempting to tackle. (numeric)
* ballCarrierName: The displayName of the ball carrier on the play (text)
* playDescription: Description of play (text)
* quarter: Game quarter (numeric)
* down: Down (numeric)
* yardsToGo: Distance needed for a first down (numeric)
* possessionTeam: Team abbr of team on offense with possession of ball (text)
* defensiveTeam: Team abbr of team on defense (text)
* yardlineSide: 3-letter team code corresponding to line-of-scrimmage (text)
* yardlineNumber: Yard line at line-of-scrimmage (numeric)
* gameClock: Time on clock of play (MM:SS)
* preSnapHomeScore: Home score prior to the play (numeric)
* preSnapVisitorScore: Visiting team score prior to the play (numeric)
* passResult: Dropback outcome of the play (C: Complete pass, I: Incomplete pass, S: Quarterback sack, IN: Intercepted pass, R: Scramble, text)
* passLength: The distance beyond the LOS that the ball traveled not including yards into the endzone. If thrown behind LOS, the value is negative. (numeric)
* penaltyYards: yards gained by offense by penalty (numeric)
* prePenaltyPlayResult: Net yards gained by the offense, before penalty yardage (numeric)
* playResult: Net yards gained by the offense, including penalty yardage (numeric)
* playNullifiedByPenalty: Whether or not an accepted penalty on the play cancels the play outcome. Y stands for yes and N stands for no. (text)
* absoluteYardlineNumber: Distance from end zone for possession team (numeric)
* offenseFormation: Formation used by possession team (text)
* defendersInTheBox: Number of defenders in close proximity to line-of-scrimmage (numeric)
* passProbability: NGS probability of next play being pass (as opposed to rush) based off model without tracking data inputs (numeric)
* preSnapHomeTeamWinProbability: The win probability of the home team before the play (numeric)
* preSnapVisitorTeamWinProbability: The win probability of the visiting team before the play (numeric)
* homeTeamWinProbabilityAdded: Win probability delta for home team (numeric)
* visitorTeamWinProbabilityAdded: Win probability delta for visitor team (numeric)
* expectedPoints: Expected points on this play (numeric)
* expectedPointsAdded: Delta of expected points on this play (numeric)
* foulName[i]: Name of the i-th penalty committed during the play. i ranges between 1 and 2 (text)
* foulNFLId[i]: nflId of the player who comitted the i-th penalty during the play. i ranges between 1 and 2 (numeric)

### Player data

* nflId: Player identification number, unique across players (numeric)
* height: Player height (text)
* weight: Player weight (numeric)
* birthDate: Date of birth (YYYY-MM-DD)
* collegeName: Player college (text)
* position: Official player position (text)
* displayName: Player name (text)

### Tackles data

* gameId: Game identifier, unique (numeric)
* playId: Play identifier, not unique across games (numeric)
* nflId: Player identification number, unique across players (numeric)
* tackle: Indicator for whether the given player made a tackle on the play (binary)
* assist: Indicator for whether the given player made an assist tackle on the play (binary)
* forcedFumble: Indicator for whether the given player forced a fumble on the play (binary)
* pff_missedTackle: Provided by Pro Football Focus (PFF). Indicator for whether the given player missed a tackle on the play (binary)

### Tracking data

Files tracking_week_[week].csv contains player tracking data from week [week].

* gameId: Game identifier, unique (numeric)
* playId: Play identifier, not unique across games (numeric)
* nflId: Player identification number, unique across players. When value is NA, row corresponds to ball. (numeric)
* displayName: Player name (text)
* frameId: Frame identifier for each play, starting at 1 (numeric)
* time: Time stamp of play (time, yyyy-mm-dd, hh:mm:ss)
* jerseyNumber: Jersey number of player (numeric)
* club: Team abbrevation of corresponding player (text)
* playDirection: Direction that the offense is moving (left or right)
* x: Player position along the long axis of the field, 0 - 120 yards. See Figure 1 below. (numeric)
* y: Player position along the short axis of the field, 0 - 53.3 yards. See Figure 1 below. (numeric)
* s: Speed in yards/second (numeric)
* a: Speed in yards/second^2 (numeric)
* dis: Distance traveled from prior time point, in yards (numeric)
* o: Player orientation (deg), 0 - 360 degrees (numeric)
* dir: Angle of player motion (deg), 0 - 360 degrees (numeric)
* event: Tagged play details, including moment of ball snap, pass release, pass catch, tackle, etc (text)
