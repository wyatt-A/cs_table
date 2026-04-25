

%% Uniform poisson disc to variable density (hamming window)

points = readcfl("~/cs_table/scatter3");

points = points * 2 - 1;

x = real(points(1,:));
y = real(points(2,:));
z = real(points(3,:));

unit_disk = sqrt(x.^2 + y.^2) < 1;
x = x(unit_disk);
y = y(unit_disk);
z = z(unit_disk);


r = linspace(0,1,1000);

rc = 0.1;
rt = 0.8;
b  = 0.1;

prob = zeros(size(r));

for i = 1:length(r)
    ri = r(i);

    if ri <= rc
        p = 1;
    elseif ri <= rt
        t = (ri - rc) / (rt - rc);   % normalize 0→1

        % Quintic smoothstep (C2 continuous)
        s = t^3 * (t * (6*t - 15) + 10);

        % invert so it goes from 1 → b
        p = (1 - b) * (1 - s) + b;
    else
        p = b;
    end

    prob(i) = p;
end

%plot(r,prob)


prob = prob / trapz(r,prob);

cdf = cumtrapz(r,prob);

plot(cdf,r);


rr = x.^2 + y.^2;

rp = interp1(cdf,r,rr,"linear");

scale = rp ./ sqrt(rr);

x = x.*scale;
y = y.*scale;


scatter3(x,y,z,".")


