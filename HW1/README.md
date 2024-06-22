# MAS - Gridworld


### Coordinate systems

0,0 ------------------->X (w)

|

|

|

|

v

Y (h)

- In config, world_size now would be: [h, w]

- 2D position in env: `pos = [x, y]`

- Convert 2D point to state number: `w * pos[1] + pos[0]`
