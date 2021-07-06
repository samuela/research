# We have to separate the tests because of https://github.com/fonsp/Pluto.jl/issues/115#issuecomment-870884010

include("boxes.jl")

import Test: @test

let b1 = Box([-1.0, -1.0], [1.0, 1.0]), b2 = Box([0.0, 0.0], [2.0, 2.0])
  # I don't want to bother defining == on Boxes.
  @test repr(intersect_boxes(b1, b2)) == "Box([0.0, 0.0], [1.0, 1.0])"
end
let b1 = Box([-1.0, -1.0], [1.0, 1.0]), b2 = Box([1.1, 1.1], [2.0, 2.0])
  @test is_empty(intersect_boxes(b1, b2))
end
let b1 = Box([-1.0, -1.0], [1.0, 1.0]), b2 = Box([0.0, 1.1], [2.0, 2.0])
  @test is_empty(intersect_boxes(b1, b2))
end
let b1 = Box([-1.0], [0.0]), b2 = Box([0.0], [2.0])
  # Two boxes that juuuuust touch have a non-empty intersection.
  @test repr(intersect_boxes(b1, b2)) == "Box([0.0], [0.0])"
  @test !is_empty(intersect_boxes(b1, b2))
end

let bs = BoxSet(Box[], Dict())
  add_box!(bs, Box([0.0, 0.0], [1.0, 1.5]))
  add_box!(bs, Box([-1.0, -1.0], [1.0, 1.0]))
  add_box!(bs, Box([-2.0, -2.0], [-1.9, -1.9]))

  @test bs.adjacency[1] == [2]
  @test bs.adjacency[2] == [1]
  @test bs.adjacency[3] == []
end
