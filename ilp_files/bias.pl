% Bias File for Popper ILP
% Defines search space for rule learning

% Target predicate to learn
head_pred(wall_following, 1).

% Predicates allowed in rule body
body_pred(signal_shape, 4).
body_pred(has_window, 2).
body_pred(scenario, 2).

% Type declarations
type(wall_following, (run,)).
type(signal_shape, (run, window, sensor, shape)).
type(has_window, (run, window)).
type(scenario, (run, scenario)).

% Direction declarations
direction(wall_following, (in,)).
direction(signal_shape, (in, in, in, out)).
direction(has_window, (in, in)).
direction(scenario, (in, in)).

% Search limits
max_clauses(5).
max_body(4).
max_vars(6).
