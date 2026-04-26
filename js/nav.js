/* Nav: hide on scroll down, show on scroll up */
(function() {
    var nav = document.querySelector('.nav-container');
    var lastY = 0;
    var threshold = 60;
    var ticking = false;

    function onScroll() {
        var y = window.scrollY;
        if (y > threshold && y > lastY) {
            nav.classList.add('nav-hidden');
        } else {
            nav.classList.remove('nav-hidden');
        }
        lastY = y;
        ticking = false;
    }

    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(onScroll);
            ticking = true;
        }
    }, { passive: true });
})();
