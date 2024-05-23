import os
BaseCase = __import__('base_case').BaseCase


class TestBoxes(BaseCase):
    @BaseCase.maybe_skip
    def test_add_boxes(self):
        self.open_main(static=True)

        self.send_input_files_from_assets([ "test_image0.jpg", "test_image1.jpg" ])
        self.click('label:contains("test_image1.jpg")')

        root_css = '[filename="test_image1.jpg"]'
        img      = self.find_element(root_css+' .input-image')
        self.click(root_css+' a.new-box')
        self.sleep(0.1)

        from selenium import webdriver
        #click and drag to create a new box
        webdriver.ActionChains(self.driver).click_and_hold(img).move_by_offset(100,100).release().perform()
        self.sleep(0.1)

        boxes = self.find_elements(root_css+' .box-overlay')
        assert len(boxes) == 1
        get_boxes_js = 'return GLOBAL.files["test_image1.jpg"].results.boxes'
        boxes0        = self.execute_script(get_boxes_js)
        assert len(boxes0) == 1 and len(boxes0[0]) == 4, boxes0
        assert self.get_text("p.box-label") == '???'

        self.hover_on_element(root_css+' .box-overlay')
        move_anchor = self.find_element(root_css+' .move-anchor')
        #click and drag to move box
        webdriver.ActionChains(self.driver).click_and_hold(move_anchor).move_by_offset(20,-20).release().perform()
        self.sleep(0.1)

        boxes1        = self.execute_script(get_boxes_js)
        #assert the box coordinates have moved
        assert  boxes1[0][0] > boxes0[0][0] \
            and boxes1[0][1] < boxes0[0][1] \
            ,   (boxes0, boxes1)
        #assert the box width/height has not changed
        assert  abs(  (boxes1[0][0] - boxes1[0][2]) - (boxes0[0][0] - boxes0[0][2])  ) < 1 \
            and abs(  (boxes1[0][1] - boxes1[0][3]) - (boxes0[0][1] - boxes0[0][3])  ) < 1 \
            ,   (boxes0, boxes1)

        
        self.hover_on_element(root_css+' .box-overlay')
        resize_anchor = self.find_element(root_css+' .resize-anchor')
        #click and drag to resize the box
        webdriver.ActionChains(self.driver).click_and_hold(resize_anchor).move_by_offset(10,0).release().perform()
        self.sleep(0.1)

        boxes2        = self.execute_script(get_boxes_js)
        #assert the box topleft has not changed, only width
        assert  abs(boxes2[0][0] - boxes1[0][0]) < 1 \
            and abs(boxes2[0][1] - boxes1[0][1]) < 1 \
            and     boxes2[0][2] > boxes1[0][2]      \
            and abs(boxes2[0][3] - boxes1[0][3]) < 1 \
            ,   (boxes1, boxes2)

        #remove box
        self.hover_on_element(root_css+' .box-overlay')
        self.click(root_css+' .close.red.icon')
        self.sleep(0.1)

        boxes = self.find_elements(root_css+' .box-overlay')
        assert len(boxes) == 0
        get_boxes_js = 'return GLOBAL.files["test_image1.jpg"].results.boxes'
        boxes3        = self.execute_script(get_boxes_js)
        assert len(boxes3) == 0, boxes3

        if self.demo_mode:
            self.sleep(1)
        return

    @BaseCase.maybe_skip
    def test_add_boxes_zoomed(self):
        self.open_main(static=True)

        self.send_input_files_from_assets([ "test_image0.jpg", "test_image1.jpg" ])
        self.click('label:contains("test_image1.jpg")')

        root_css = '[filename="test_image1.jpg"]'
        img      = self.find_element(root_css+' .input-image')
        self.click(root_css+' a.new-box')
        self.sleep(0.1)

        #zoom in
        script_zoom   = f''' $('{root_css} .transform-box')[0].dispatchEvent( new WheelEvent("wheel", {{deltaY:-10, shiftKey:true}}) ) '''
        for i in range(20):
            self.execute_script(script_zoom)

        from selenium import webdriver
        #click and drag to create a new box
        webdriver.ActionChains(self.driver).click_and_hold(img).move_by_offset(100,100).release().perform()
        self.sleep(0.1)

        get_boxes_js = 'return GLOBAL.files["test_image1.jpg"].results.boxes'
        box0         = self.execute_script(get_boxes_js)[0]
        #should be strictly within image bounds
        assert 0 < box0[0] < 512 and 0 < box0[1] < 512, box0

