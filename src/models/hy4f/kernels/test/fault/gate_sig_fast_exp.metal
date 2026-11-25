        // the planted fault (a red check, never served): the sigmoid's exp on the fast path
        const float ye = 1 / (1 + metal::fast::exp(metal::abs(yv)));
        const float sg = (yv < 0) ? ye : 1 - ye;
