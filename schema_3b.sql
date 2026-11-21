-- Lesson 3b: structured tables for lookups, stats and the name check.
-- Run once in Supabase: SQL Editor -> paste -> Run. (The chunks table stays as it is.)

-- One row per match: what the router uses to work out "which match?".
create table if not exists matches (
  match_id        text primary key,           -- Cricsheet file name, same as chunks.match_id
  season          text not null,
  match_date      date not null,
  match_number    int,                        -- 1-70 league; playoffs numbered 71-74 by date
  stage           text not null,              -- 'League', 'Qualifier 1', 'Eliminator', 'Qualifier 2', 'Final'
  team1           text not null,
  team2           text not null,
  venue           text,
  winner          text,
  result          text,                       -- e.g. 'Royal Challengers Bengaluru won by 6 runs'
  player_of_match text[]
);

-- One row per ball: what the stats route counts. Answers that must be
-- CALCULATED (most sixes, economy, a player's season) come from here, not from RAG.
create table if not exists deliveries (
  match_id       text not null references matches(match_id),
  season         text not null,
  innings        int  not null,
  seq            int  not null,               -- order of the ball within the innings (wides/no-balls included)
  over_number    int  not null,               -- 1-based, like the chunks (over 20 = the last over)
  ball_label     text not null,               -- cricket notation, e.g. '19.3'
  phase          text,                        -- powerplay / middle / death
  batting_team   text not null,
  bowling_team   text not null,
  batter         text not null,
  bowler         text not null,
  non_striker    text,
  batter_runs    int  not null,
  extra_runs     int  not null,
  total_runs     int  not null,
  bowler_runs    int  not null,               -- runs charged to the bowler (no byes / leg byes)
  is_legal       boolean not null,            -- false for wides and no-balls
  is_wide        boolean not null,            -- wides don't count as a ball faced by the batter
  is_four        boolean not null,
  is_six         boolean not null,
  is_wicket      boolean not null,
  wicket_kind    text,
  player_out     text,
  fielder        text,
  bowler_wicket  boolean not null,            -- the bowler gets credit (not run outs etc.)
  primary key (match_id, innings, seq)
);

-- Everyone who played in a season, for the name check ("is this player in our data?").
create table if not exists players (
  season        text not null,
  name          text not null,               -- as Cricsheet writes it, e.g. 'V Kohli'
  cricsheet_id  text,
  teams         text[],
  primary key (season, name)
);

alter table matches    enable row level security;
alter table deliveries enable row level security;
alter table players    enable row level security;

create index if not exists deliveries_season_batter on deliveries (season, batter);
create index if not exists deliveries_season_bowler on deliveries (season, bowler);
create index if not exists deliveries_match         on deliveries (match_id, innings, over_number);
create index if not exists matches_season_stage     on matches (season, stage, match_number);
