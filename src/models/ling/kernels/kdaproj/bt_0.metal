    // DEC-ONE's statements at the threadgroup's member's tiles: BTC member 0's (BT0) here, members 1 and 2's (BT1) in
    // the other branch (one branch when BT0 == BT1). The branch is the threadgroup's own, so every thread takes it
    if (jm == 0u || BT0 == BT1) {
    constexpr uint BTC = uint(BT0);
